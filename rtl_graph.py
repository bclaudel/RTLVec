"""Versioned graphs of Yosys word-level cells connected through signal bits."""
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import scipy
from scipy.sparse import coo_matrix, diags, eye
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
import torch
from pyg_compat import Data


NODE_TYPES = (
    "input", "output", "signal", "const0", "const1", "constx", "constz",
    "$not", "$pos", "$neg", "$and", "$or", "$xor", "$xnor",
    "$reduce_and", "$reduce_or", "$reduce_xor", "$reduce_xnor", "$reduce_bool",
    "$logic_not", "$logic_and", "$logic_or", "$shl", "$shr", "$sshl", "$sshr",
    "$shift", "$shiftx", "$lt", "$le", "$eq", "$ne", "$eqx", "$nex", "$ge", "$gt",
    "$add", "$sub", "$mul", "$div", "$mod", "$divfloor", "$modfloor", "$pow",
    "$mux", "$pmux", "$bmux", "$demux", "$slice", "$concat",
    "$dff", "$adff", "$sdff", "$dffe", "$adffe", "$sdffe", "$sdffce",
    "$aldff", "$aldffe", "$dffsr", "$dffsre", "$dlatch", "$adlatch", "$dlatchsr", "$sr",
)
TYPE_IDS = {name: i for i, name in enumerate(NODE_TYPES)}
EDGE_ROLES = ("data", "control", "clock", "reset", "state")
PASSES = ("proc", "flatten", "opt_clean", "check -assert")
GRAPH_SPEC = {
    "name": "structrtl-yosys-v1",
    "node_types": list(NODE_TYPES),
    "edge_roles": list(EDGE_ROLES),
    "signals": "one node per used bit; aliases share identity",
    "constants": "shared nodes for 0/1/x/z",
    "widths": "port length or maximum cell output length; signal/constants=1",
    "edges": "directed pin connections, with role and port/bit identity",
    "passes": list(PASSES),
    "positions": "16 positive eigenmodes of undirected symmetric-normalized Laplacian; zero-pad",
    "position_sign": "largest-magnitude entry nonnegative",
    "unsupported": "memories, inout, blackboxes, unknown cells, undriven/multiply-driven signals",
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def preprocessing_contract(yosys_version):
    return {"graph_spec": GRAPH_SPEC, "yosys_version": yosys_version,
            "extractor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "conversion_sha256": hashlib.sha256(Path(__file__).with_name("prepare_data.py").read_bytes()).hexdigest(),
            "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "networkx_version": nx.__version__}


def positional_encoding(n, edges, k=16):
    """Direction is preserved in edge_index, but ignored for spectral coordinates."""
    rows, cols = edges
    adjacency = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.data[:] = 1
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv = np.zeros(n)
    inv[degree > 0] = degree[degree > 0] ** -0.5
    laplacian = eye(n, format="csr") - diags(inv) @ adjacency @ diags(inv)
    # Isolated vertices have zero Laplacian rows, not artificial eigenvalue 1.
    laplacian = laplacian - diags((degree == 0).astype(float))
    if n <= 256:
        values, vectors = np.linalg.eigh(laplacian.toarray())
    else:
        components = connected_components(adjacency, directed=False, return_labels=False)
        count = min(n - 1, k + components)
        values, vectors = eigsh(laplacian, k=count, which="SM", tol=1e-8,
                                v0=np.random.default_rng(42).normal(size=n))
    order = np.argsort(values)
    selected = [i for i in order if values[i] > 1e-8][:k]
    result = np.zeros((n, k), dtype=np.float32)
    for col, index in enumerate(selected):
        vector = vectors[:, index]
        if vector[np.argmax(np.abs(vector))] < 0:
            vector = -vector
        result[:, col] = vector
    return torch.from_numpy(result)


def cell_role(port, direction):
    if direction == "output":
        return 4 if port == "Q" else 0
    if port in ("CLK", "C"):
        return 2
    if port in ("ARST", "SRST", "CLR", "SET", "R"):
        return 3
    if port in ("S", "EN", "E", "ALOAD"):
        return 1
    return 0


def graph_from_json(netlist, top, *, max_nodes=2048):
    modules = netlist["modules"]
    if top not in modules:
        raise ValueError(f"Top module {top!r} is absent from Yosys output")
    module = modules[top]
    if module.get("memories"):
        raise ValueError("Memories are not supported by graph schema v1")
    ports, cells = module.get("ports", {}), module.get("cells", {})
    nodes, widths, names, parameters, pin_edges = [], [], [], [], set()
    def add_node(kind, width, name, params=None):
        if kind not in TYPE_IDS:
            raise ValueError(f"Unsupported cell/node type: {kind}")
        if width < 1:
            raise ValueError(f"Zero-width node: {name}")
        if len(nodes) >= max_nodes:
            raise ValueError(f"Graph exceeds max_nodes={max_nodes}; reduce design size or raise the limit")
        nodes.append(TYPE_IDS[kind]); widths.append(width); names.append(name)
        parameters.append(params or {})
        return len(nodes) - 1
    used_bits = set()
    for port in ports.values():
        if port["direction"] not in ("input", "output"):
            raise ValueError("Inout ports are not supported by graph schema v1")
        used_bits.update(port["bits"])
    for cell in cells.values():
        if cell["type"] == "$scopeinfo":
            continue
        if cell["type"] not in TYPE_IDS:
            raise ValueError(f"Unsupported cell type {cell['type']}; memories/blackboxes are not supported")
        for connection in cell["connections"].values():
            used_bits.update(connection)
    signal_nodes = {bit: add_node("signal", 1, f"bit:{bit}")
                    for bit in sorted(b for b in used_bits if isinstance(b, int))}
    for bit in sorted(b for b in used_bits if isinstance(b, str)):
        if bit not in ("0", "1", "x", "z"):
            raise ValueError(f"Unknown constant {bit!r}")
        signal_nodes[bit] = add_node(f"const{bit}", 1, f"constant:{bit}")
    drivers = {bit: set() for bit in signal_nodes if isinstance(bit, int)}
    for name, port in sorted(ports.items()):
        node = add_node(port["direction"], len(port["bits"]), f"port:{name}")
        for offset, bit in enumerate(port["bits"]):
            signal = signal_nodes[bit]
            if port["direction"] == "input":
                if not isinstance(bit, int):
                    raise ValueError("Input ports cannot be tied to constants")
                drivers[bit].add(node)
                pin_edges.add((node, signal, 0, f"bit:{offset}"))
            else:
                pin_edges.add((signal, node, 0, f"bit:{offset}"))
    for name, cell in sorted(cells.items()):
        if cell["type"] == "$scopeinfo":
            continue
        directions = cell.get("port_directions", {})
        if set(directions) != set(cell["connections"]):
            raise ValueError(f"Unknown pin directions for {name}")
        if any(direction not in ("input", "output") for direction in directions.values()):
            raise ValueError(f"Unsupported inout pin on {name}")
        outputs = [len(cell["connections"][pin]) for pin, direction in directions.items() if direction == "output"]
        if not outputs:
            raise ValueError(f"Cell {name} has no outputs")
        node = add_node(cell["type"], max(outputs), f"cell:{name}", cell.get("parameters", {}))
        for pin, connection in sorted(cell["connections"].items()):
            direction = directions[pin]
            role = cell_role(pin, direction)
            for offset, bit in enumerate(connection):
                signal = signal_nodes[bit]
                if direction == "output":
                    if not isinstance(bit, int):
                        raise ValueError(f"Cell {name} drives a constant")
                    drivers[bit].add(node)
                    pin_edges.add((node, signal, role, f"{pin}:{offset}"))
                else:
                    pin_edges.add((signal, node, role, f"{pin}:{offset}"))
    if any(len(values) != 1 for values in drivers.values()):
        raise ValueError("Undriven or multiply-driven signal bits are not supported")
    if len(nodes) < 2 or not pin_edges:
        raise ValueError("Graph must contain at least two nodes and one edge")
    ordered = sorted(pin_edges)
    edge_index = torch.tensor([(u, v) for u, v, _, _ in ordered], dtype=torch.long).t().contiguous()
    labels = torch.tensor(nodes, dtype=torch.long)
    graph = Data(x=torch.cat([torch.nn.functional.one_hot(labels, len(NODE_TYPES)).float(),
                              torch.tensor(widths, dtype=torch.float32)[:, None]], dim=1),
                 y=labels[:, None], edge_index=edge_index,
                 edge_attr=torch.tensor([role for _, _, role, _ in ordered], dtype=torch.long)[:, None],
                 pos_enc=positional_encoding(len(nodes), edge_index.numpy()))
    graph.edge_ports = [pin for _, _, _, pin in ordered]
    graph.node_names = names
    graph.node_parameters = parameters
    graph.graph_schema = GRAPH_SPEC["name"]
    return graph


def typed_graph(graph):
    """Exact deduplication ignores naming and positions, but retains ordered pins."""
    result = nx.DiGraph()
    for i, (kind, width, parameters) in enumerate(zip(
            graph.y[:, 0].tolist(), graph.x[:, -1].tolist(), graph.node_parameters)):
        result.add_node(i, label=f"{int(kind)}:{int(width)}:{fingerprint(parameters)}")
    grouped = {}
    for pair, role, pin in zip(graph.edge_index.t().tolist(), graph.edge_attr[:, 0].tolist(), graph.edge_ports):
        grouped.setdefault(tuple(pair), []).append((int(role), pin))
    for (u, v), pins in grouped.items():
        result.add_edge(u, v, pins=json.dumps(sorted(pins)))
    return result


def typed_graph_hash(graph):
    return nx.weisfeiler_lehman_graph_hash(typed_graph(graph), node_attr="label", edge_attr="pins")


def graphs_equal(left, right):
    from graph_isomorphism import attributed_graphs_equal
    return attributed_graphs_equal(typed_graph(left), typed_graph(right))
