# Stage 1 with DeepCircuitX

The importer uses the repository URL list published on the authors'
[Source RTL code page](https://zeju.gitbook.io/lcm-team/deepcircuitx/souce-rtl-code).
By default it reads the module-level and encoder/decoder lists; `--collections
all` also includes chip, IP and RISC-V lists. It deduplicates repository URLs
and selects a deterministic prefix spread across functional categories. It does
not download or use the GPT annotations, existing graphs or physical labels.

## Build a corpus

```bash
uv sync --locked
uv run python import_deepcircuitx.py --repos 200 --jobs 8
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python prepare_data.py convert \
  --manifest data/deepcircuitx/projects.json \
  --output data/deepcircuitx/generated --jobs 8 --timeout 30
uv run python prepare_data.py assemble \
  --input data/deepcircuitx/generated --output data/deepcircuitx/stage1 \
  --allow-rejected
uv run python prepare_data.py inspect --data-dir data/deepcircuitx/stage1
```

Conversion exits with status 1 when any candidate is rejected. This is expected
for this collection: check `generated/conversion.json`, then run assembly as a
separate command. `--allow-rejected` explicitly selects successful conversions;
it never suppresses their graph validation or deletes the rejection report.
The default assembly behavior still requires all candidates to pass.

Downloads require network access. The URL-list archive is cached locally, and
existing repository checkouts retain their original pinned commit on reruns.
Increase `--repos` to expand the same prefix. The default cap is 40 candidate
modules per repository, adjustable with `--modules-per-repo`. A failed partial
checkout from older importer versions must be moved aside before retrying;
new failed downloads are discarded, and failures are listed in `import.json`.
The raw sources, graphs and datasets live in the ignored `data/deepcircuitx/` tree.

To attempt every repository in all published source collections and discover
all candidate modules within each checkout:

```bash
uv run python import_deepcircuitx.py \
  --collections all --repos 0 --modules-per-repo 0 --jobs 16
```

The cached index contains 4,547 unique repository URLs across module,
encoder/decoder, chip, IP and RISC-V lists. Zero disables the respective cap;
it does not disable source-language, dependency or graph-validation checks.
Failed new downloads are discarded automatically so a later run can retry.
`import-progress.json` records progress every 50 repositories; `import.json`
and `projects.json` are published when discovery finishes. Repositories listed
under several collections are fetched once. Existing checkouts retain their
original pinned commits.

If Git transport fails for public repositories, retry missing checkouts with
source archives:

```bash
uv run python import_deepcircuitx.py --collections all --repos 0 \
  --modules-per-repo 0 --jobs 16 --transport archive
```

GitHub tar archives identify the exact source commit in their global header.
Archive snapshots store that commit, the download URL and the compressed
archive SHA-256 in `.deepcircuitx-source.json`. They retain regular files and
skip symbolic/hard links. Invalid paths and unidentified commits are rejected.
Archives are limited to 512 MB compressed and 2 GB expanded; excessive or
unavailable repositories are reported. Existing Git and archive snapshots are
reused without updates. `--local-only` rebuilds the candidate manifest from
existing snapshots without attempting network downloads. The one-time
`download-results.json` records direct Git retries made during the full import.

Then repeat conversion and assembly from the commands above. Assembly into the
same `stage1` directory preserves existing repository-family split assignments
and prioritizes existing graph representatives during deduplication. If the
converter/toolchain changes, choose a new dataset directory instead. Importing
and rebuilding the dataset does not start training.

For the full import in this workspace, conversion uses the locally installed
[OSS CAD Suite 2026-10-08](https://github.com/YosysHQ/oss-cad-suite-build/releases/tag/2026-10-08)
native Yosys bundle. Its release URL, archive SHA-256 and Yosys version are
recorded in `data/deepcircuitx/tools/toolchain.json`. The four included fixtures
produce structurally equivalent graphs with native Yosys and the portable
YoWASP toolchain. Their preprocessing fingerprints remain distinct.
The exact converter sources that produced the full corpus are preserved under
`generated-full/producer/`, with hashes matching its preprocessing contract.
The later exact-matching performance improvements preserve the graph schema
and typed-graph hashes. Assembly validates every stored graph against the
original conversion report.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python prepare_data.py convert \
  --manifest data/deepcircuitx/projects.json \
  --output data/deepcircuitx/generated-full --jobs 16 --timeout 30 \
  --yosys data/deepcircuitx/tools/oss-cad-suite/bin/yosys
OMP_NUM_THREADS=1 uv run python prepare_data.py assemble \
  --input data/deepcircuitx/generated-full --output data/deepcircuitx/stage1-full \
  --allow-rejected --split-from data/deepcircuitx/stage1
uv run python prepare_data.py inspect --data-dir data/deepcircuitx/stage1-full
```

`--split-from` seeds a new dataset's repository assignments and deduplication
representative preferences from a validated reference dataset. It never copies
reference graph tensors into the new dataset, so it also works when the
toolchain changes. Later assemblies reuse the new dataset's own assignments.
The original 635-graph dataset remains at `stage1/`; the full corpus uses
`stage1-full/`. A copy of the original dataset and experiment report also lives
under `snapshots/pilot-635/`. Training on the expanded corpus must explicitly
select `--data-dir data/deepcircuitx/stage1-full`.

## Full import (2026-10-09)

All 4,547 unique repository URLs in the published source index were attempted.
The importer downloaded 4,356 repositories and discovered 87,609 candidate
modules without per-repository caps. The 191 download failures include 180
unavailable repositories (HTTP 404), ten archives exceeding the size limits,
and one unsupported filename length. Sources occupy approximately 116 GiB.

Conversion accepted 37,020 candidates and rejected 50,589 with recorded reasons.
Exact assembly removed 16,833 duplicates, yielding **20,187 graphs from 2,869
repository families**, about 32 times the original 635-graph corpus:

| Split | Graphs | Nodes | Directed pin edges |
| --- | ---: | ---: | ---: |
| Training | 16,403 | 6,485,617 | 17,781,914 |
| Validation | 3,784 | 1,366,462 | 3,779,714 |

The ready-to-load files are `stage1-full/train.pt`, `stage1-full/val.pt` and
`stage1-full/dataset.json`, approximately 3.7 GiB together. Full conversion
artifacts occupy approximately 18 GiB. Original data and its family assignments
are preserved. `expansion.json` records counts, fingerprints, toolchain and
verification results; `rejection-summary.json` groups conversion failures.
The complete dataset loader passed, and a fresh full-size encoder produced
finite 512-dimensional node embeddings for 2-, 205- and 2,048-node graphs.
All 39 tests passed. Training on the expanded corpus has not been started.
The full import includes the source collections, not the authors' annotations
or physical labels, and only supported, validated graphs enter the dataset.

## Train

```bash
OMP_NUM_THREADS=4 uv run python train.py \
  --data-dir data/deepcircuitx/stage1-full --output-dir saves/deepcircuitx-full \
  --num_epochs 3 --max_nodes_per_batch 2048 --transformer_num_heads 4 \
  --stratified_masking --random_flip_posenc --learning_rate 2e-5
```

This starts a new encoder using the full default 8-layer GIN and 8-layer
Transformer. Three epochs check execution and early learning; they do not
establish convergence or downstream area/delay prediction quality. Best and last
checkpoints, model arguments, the dataset manifest and per-epoch metrics are
saved in a unique run directory. Longer training can use the same dataset with
a larger `--num_epochs` value; checkpoint resumption is not implemented.

## First measured run (2026-10-09)

The first 200 repository entries yielded 191 downloads and 1,333 candidate
modules. Conversion accepted 712 candidates; assembly removed 77 duplicates,
leaving **635 graphs from 140 repository families**. The grouped split contains
485 training graphs and 150 validation graphs, totaling 138,145 nodes and
359,316 directed pin edges. Graph sizes range from 3 to 2,046 nodes, with a median
of 84. Training covers 43 of the 65 fixed node types.

Three epochs on the RTX 5090 completed successfully:

| Epoch | Training loss | Validation masked-node accuracy | Validation edge accuracy |
| --- | --- | --- | --- |
| 1 | 0.7804 | 2.51% | 63.23% |
| 2 | 0.6474 | 4.91% | 81.84% |
| 3 | 0.5656 | 29.89% | 81.20% |

These are the existing training script's metrics with randomly sampled masks
and edges. The best checkpoint is selected by the sum of the two validation
accuracies. The corpus and checkpoint use our own graph schema, so these values
are not directly comparable to the paper's reported results.

The local `data/deepcircuitx/experiment.json` records dataset fingerprints,
metrics and checkpoint paths. Per-candidate failures remain in
`data/deepcircuitx/generated/conversion.json`. The main rejected groups were
memories/external memory files (134), invalid or undriven signals (117), missing
dependencies/technology cells (116), and excessive graph size (62).

## Selection and provenance

Discovery finds ordinary named Verilog modules, resolves known module instances
and literal includes, and skips testbench/simulation directories. Ambiguous
module definitions and unsafe/missing include paths are reported. This is a
conservative text-based dependency finder, not an HDL parser: macro-dependent
dependencies and project-specific build configurations can require a manually
curated manifest. Upstream build scripts are never executed.

Each design records the upstream URL, pinned commit, DeepCircuitX categories and
root license-file paths. License files remain in the checkouts; a missing entry
means no root license file was found, not permission to redistribute the source.
All selected modules from a repository share one `project_family`. Assembly
removes exact typed-graph duplicates and assigns whole repository families to
training or validation. Matching uses joint vertex-color refinement to prune
candidates, then exact Nauty matching with each directed edge encoded as a
colored tail/head path. The colors preserve direction and every pin/role
attribute. A shared graph hash alone never establishes equality. Repository
grouping does not automatically identify
non-identical forks; resolving upstream fork relationships is a further step
before making strong generalization claims.

Our existing graph limits apply: plain Verilog, supported Yosys operation cells,
no memories/inout/blackboxes, and at most 2048 nodes by default. Raising
`--max-nodes` also requires an adequate training node budget and increases
attention memory costs. Non-UTF8 comments are tolerated during include checks;
the source bytes and hashes are preserved.

Accepted means conversion and structural validation succeeded. We have not yet
implemented the paper's Verilator meaningful-output filter or generated
technology-mapped area/delay labels. See [preprocessing.md](preprocessing.md)
for the graph representation and stage-2 boundary.
