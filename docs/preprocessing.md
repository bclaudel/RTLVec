# Our RTL graph pipeline

## Design choice and relation to the paper

We own the representation and do not require backward compatibility with the
supplied StructRTL graphs or checkpoints. The paper's extraction uses an AST
traversal with Stagira. Our extractor consumes Yosys JSON directly, after
behavioral processes are lowered into word-level operations. This is a
StructRTL-inspired approach, not a reproduction of the paper's preprocessing.

```text
Verilog + explicit project configuration
    → Yosys hierarchy elaboration and process lowering
    → word-level cells connected through signal bits
    → graph tensors and spectral positions
    → grouped train/validation corpus
    → fresh stage-1 encoder
```

## Input manifest

See `examples/projects.json`. `schema_version` is 1, and `designs` is a nonempty
list. Each entry requires `design_id`, `project_family`, `source_provenance`,
`sources`, and `top`. Optional `include_dirs`, `defines`, and `parameters` default
to empty collections. Sources and include paths are relative to the manifest;
absolute paths also work. Sources are synthesizable `.v` files (`.vh` files may
also be explicitly listed).

```json
{
  "schema_version": 1,
  "designs": [{
    "design_id": "alu-width16-v1",
    "project_family": "my-alu",
    "source_provenance": "repository URL and commit",
    "sources": ["rtl/alu.v", "rtl/arithmetic.v"],
    "top": "alu",
    "include_dirs": ["rtl/include"],
    "defines": {"ENABLE_MULTIPLY": 1},
    "parameters": {"WIDTH": 16}
  }]
}
```

Use the same family for related variants and designs from the same source
project. IDs must be unique. Top modules and define/parameter names use ordinary
Verilog identifiers. Define and parameter values support simple numeric/macro
expressions, not arbitrary Yosys commands or quoted string values.

`validate` checks metadata and file availability; only `convert` parses and
elaborates RTL. Conversion snapshots declared sources, adjacent `.vh` files,
and declared include directories, checking their hashes. Use include directories
for other headers; undeclared dependencies and absolute RTL includes are
unsupported. All files within declared include directories affect cache identity,
so keep these directories focused on compilation inputs.

## Lowering and extraction contract

Graph schema `structrtl-yosys-v1` is defined by `GRAPH_SPEC` in `rtl_graph.py`.

The fixed Yosys script reads sources with their include paths and defines,
applies parameter overrides, and runs:

```text
hierarchy -check -top <top>
proc
flatten
opt_clean
check -assert
write_json <output>
```

No technology mapping, ABC, arithmetic decomposition, FSM extraction or memory
mapping is performed. Addition remains an `$add` operation rather than a
collection of technology-library gates. Control becomes mux/select and enable
connections, and storage becomes flip-flop/latch cells. This graph is an RTL
operation/signal dependency representation rather than a source AST.

Node ordering is deterministic: used signal bit IDs, constant symbols, named
module ports, then named cells. Aliased wires share the same signal bit node.

| Node kind | Width |
|---|---|
| Input/output port | Number of connected bits |
| Signal bit | 1 |
| Constant 0, 1, x or z | 1 |
| Word-level cell | Maximum output-port width |

Constants share one node per symbol. Cells use a fixed numerical vocabulary,
listed in order in `NODE_TYPES`. Unsupported cells are rejected, never mapped
silently into an unknown category. The vocabulary covers common arithmetic,
logic, comparisons, shifts, multiplexers and registers/latches.

Connections produce directed pin edges: input port → signal, signal → cell,
cell → signal, and signal → output port. Pin names and bit indices preserve
operand order. The roles are data=0, control=1, clock=2, reset=3, state=4. Inout,
memories, unknown/blackbox cells, and undriven or multiply-driven signals are
rejected. Sequential feedback is retained; cycles through registers are valid.

The tensor format is:

| Field | Meaning |
|---|---|
| `x` | float32 `[N, number_of_types + 1]`: type one-hot plus positive width |
| `y` | int64 `[N, 1]`: node-type targets |
| `edge_index` | int64 `[2, E]`: directed pin connections |
| `edge_attr` | int64 `[E, 1]`: connection roles |
| `pos_enc` | float32 `[N, 16]`: graph structural coordinates |
| `edge_ports` | Pin/bit identity strings, one per edge |
| `node_parameters` | Yosys cell parameters, including signedness |

The current GIN/GAT encoder consumes node types, widths, directed connectivity
and positions. Edge roles, pin strings and cell parameters are preserved for
validation, exact deduplication and future models; they are not directly embedded
by the current encoder. Repeated pin connections appear as parallel edges.

Positions use the undirected projection of the directed graph, with binary
adjacency and no self-loops, and a symmetric normalized Laplacian. All zero modes
are omitted. Up to 16 positive eigenmodes are selected, signs are canonicalized
using the largest-magnitude entry, and missing columns are zero-padded. Isolated
vertices have zero Laplacian rows. Small graphs use dense `eigh`; larger graphs
use sparse `eigsh` with a fixed initial vector. Random sign flips remain an
optional training augmentation. Degenerate eigenspaces can depend on numerical
libraries; their versions and the extractor code hashes are fingerprinted.

## Conversion, caching and failure reports

```bash
uv run python prepare_data.py validate --manifest examples/projects.json
uv run python prepare_data.py convert --manifest examples/projects.json --output data/generated
```

Default per-design timeout is 120 seconds and maximum graph size is 2048 nodes.
Override with `--timeout` and `--max-nodes`; optionally select `--yosys`.
Limits keep bit-expanded graphs and dense Transformer attention manageable.

Conversion writes `conversion.json`, per-design logs, cached graph files, and
Yosys JSON netlists. Cache keys include design/family identity, source and header
hashes, configuration, and preprocessing fingerprint. A cache hit also verifies
the artifact's content hash. Changed inputs or converter code invalidate it.
Old cache files are retained but only current report entries are assembled.

Failures are recorded individually while other designs continue. Conversion
returns nonzero when any design fails. Fix rejected entries and rerun conversion;
assembly refuses reports containing failures. First use of YoWASP may take extra
time while its WebAssembly cache is compiled.

## Dataset assembly and splits

```bash
uv run python prepare_data.py assemble --input data/generated --output data/expanded
uv run python prepare_data.py inspect --data-dir data/expanded
```

Assembly removes duplicate source/configuration inputs and exact typed graphs.
A structural hash narrows candidates; graph isomorphism verifies equality using
node types, widths, cell parameters, and directed ordered-pin connections.
Names and spectral coordinates do not affect duplicate detection. Duplicate IDs
are recorded, and the conversion report retains their source provenance.

The first assembly assigns approximately 20% of unique families to validation,
using deterministic seed-42 hash ordering, with at least one family in each
split. Subsequent assemblies preserve every saved assignment and hash new
families independently at a 20% validation threshold. At least two distinct,
nonduplicate families and nonempty splits are required.

Assembly rebuilds the corpus from the current conversion report; it does not
implicitly append cached files. Keep all desired designs in the manifest.
Outputs are `train.pt`, `val.pt`, and `dataset.json`, which records design IDs,
source hashes, provenance, family assignments, class counts, file hashes,
preprocessing identity, and a dataset fingerprint. Training verifies these before
loading the corpus. Files are staged and the manifest is published last; an
interrupted publication is detected by content hashes and can be repaired by
rerunning assembly. Changing preprocessing requires a new output directory.
The original files in `data/` are preserved and rejected by the new loader.

## Training and stage-2 reuse

```bash
uv run python train.py --data-dir data/expanded --num_epochs 1 \
  --transformer_num_heads 4 --stratified_masking --random_flip_posenc \
  --learning_rate 2e-5
```

The type count is derived from the fixed graph vocabulary. Widths are enabled
by default and embedded by an MLP from `log2(1 + width)`, without the old 32-bit
cap. Both training objectives share this preprocessing. `--delete_node_width_embedding`
is available for ablations. Classes absent from training receive zero
class-balanced weights; aggregate accuracy still includes all validation targets.

Size-aware batching is enabled by default, bounded by both `--batch_size` and
`--max_nodes_per_batch` (4096). Use `--no-size-aware-batching` to use only a graph
count limit. The node budget must fit the largest graph. Loss averages count
actual batches rather than an estimated sampler length.

Runs save `args.json`, `dataset.json`, `metrics.jsonl`, `best.pt`, and `last.pt`
below `saves/joint/` or `--output-dir`. Schema-2 checkpoints contain model settings,
weights, optimizer/scheduler state, fingerprints and metrics. Original/checkpoints
from the earlier compatibility-gated version are unsupported. Training resume,
physical regression and distillation remain later work.

```python
import torch
from checkpoint import load_encoder

encoder = load_encoder(
    "saves/joint/<run>/best.pt",
    preprocessing_fingerprint=dataset_metadata["preprocessing_fingerprint"],
)
with torch.no_grad():
    embeddings = encoder.encode_nodes(batch.x, batch.edge_index, batch.batch, batch.pos_enc)
```

Stage 2 can join physical labels to the retained design IDs and sources, then
pool these embeddings into circuit vectors. No synthesis labels are fabricated
for stage 1. The four checked-in examples demonstrate functionality; meaningful
representation learning requires a larger and more varied corpus.
