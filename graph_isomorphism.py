"""Exact directed, attributed graph matching with invariant candidate pruning."""
from collections import Counter

import networkx as nx
import pynauty


def attributed_graphs_equal(left, right):
    if (len(left), left.number_of_edges()) != (len(right), right.number_of_edges()):
        return False
    if not left:
        return True
    # Repeated vendor modules often have identical vertex numbering as well as
    # identical attributes. This is already an explicit isomorphism witness.
    if nx.utils.graphs_equal(left, right):
        return True
    if Counter(map(len, nx.weakly_connected_components(left))) != Counter(map(len, nx.weakly_connected_components(right))):
        return False
    # Joint color refinement gives equivalent vertices identical colors, without
    # depending on vertex numbering. Colors are necessary invariants, not proof
    # of isomorphism; the final match remains exact.
    graphs = (left, right)
    colors = [{u: data['label'] for u, data in g.nodes(data=True)} for g in graphs]
    for _ in range(16):
        signatures = []
        for g, current in zip(graphs, colors):
            signatures.append({u: (current[u],
                tuple(sorted((current[v], g[v][u]['pins']) for v in g.predecessors(u))),
                tuple(sorted((current[v], g[u][v]['pins']) for v in g.successors(u)))) for u in g})
        palette = {signature: i for i, signature in enumerate(dict.fromkeys(
            signature for values in signatures for signature in values.values()))}
        refined = [{u: palette[value] for u, value in values.items()} for values in signatures]
        if Counter(refined[0].values()) != Counter(refined[1].values()):
            return False
        stable = all(len(set(old.values())) == len(set(new.values())) for old, new in zip(colors, refined))
        colors = refined
        if stable:
            break
    # Encode each directed edge as u--tail--head--v. Distinct tail/head colors
    # preserve direction while allowing Nauty's faster undirected algorithm.
    # Pin/role attributes are retained in both gadget vertex colors.
    expanded = []
    for g, current in zip(graphs, colors):
        result = nx.Graph()
        for u, data in g.nodes(data=True):
            result.add_node(('node', u), label=('node', data['label'], current[u]))
        for i, (u, v, data) in enumerate(g.edges(data=True)):
            tail, head = ('tail', i), ('head', i)
            result.add_node(tail, label=('tail', data['pins']))
            result.add_node(head, label=('head', data['pins']))
            result.add_edges_from([(('node', u), tail), (tail, head), (head, ('node', v))])
        expanded.append(result)
    label_counts = [Counter(nx.get_node_attributes(g, 'label').values()) for g in expanded]
    if label_counts[0] != label_counts[1]:
        return False
    palette = sorted(label_counts[0])
    nauty_graphs = []
    for g in expanded:
        indices = {u: i for i, u in enumerate(g)}
        classes = {label: set() for label in palette}
        for u, data in g.nodes(data=True):
            classes[data['label']].add(indices[u])
        nauty_graphs.append(pynauty.Graph(len(g), directed=False,
            adjacency_dict={indices[u]: [indices[v] for v in g.neighbors(u)] for u in g},
            vertex_coloring=[classes[label] for label in palette]))
    return pynauty.isomorphic(*nauty_graphs)
