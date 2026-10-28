# StructRTL

Self-supervised RTL graph learning, inspired by
[Beyond Tokens: Enhancing RTL Quality Estimation via Structural Graph Learning](https://arxiv.org/abs/2508.18730).
This checkout now uses its own versioned Yosys-based graph representation and
trains new encoders from scratch. The authors' original graph files remain in
`data/` as reference artifacts; they are not inputs to the new pipeline.

## Quick start

```bash
uv sync --locked
uv run python prepare_data.py validate --manifest examples/projects.json
uv run python prepare_data.py convert --manifest examples/projects.json --output data/generated
uv run python prepare_data.py assemble --input data/generated --output data/expanded
uv run python train.py --data-dir data/expanded --num_epochs 1 \
  --transformer_num_heads 4 --stratified_masking --random_flip_posenc \
  --learning_rate 2e-5
```

The four included examples exercise combinational logic, selection, registers,
feedback, and multi-file hierarchy. They are a smoke-test corpus, not enough data
to establish useful generalization. Add your projects to a manifest using the
same format, then repeat conversion and assembly.

PyTorch Geometric currently calls TorchScript while importing unused pooling
classes, producing a Python 3.14 compatibility warning. `pyg_compat.py` filters
only that specific warning during the dependency import, pending the
[upstream fix](https://github.com/pyg-team/pytorch_geometric/issues/10697).
It does not migrate or disable TorchScript; warnings from later TorchScript
calls and other dependency warnings remain visible.

For a larger corpus, [the DeepCircuitX guide](docs/deepcircuitx.md) describes
downloading real RTL projects, importing modules, filtering conversion failures,
and running stage 1 training. Start with:

```bash
uv run python import_deepcircuitx.py --repos 200 --jobs 8
```

Yosys is supplied by the locked `yowasp-yosys` dependency, so a system install is
not required. The converter prefers a native `yosys` if available; use `--yosys`
to select an executable explicitly. Each toolchain produces a recorded
preprocessing fingerprint. Use a new output directory when changing toolchains.

Training learns node types and graph connectivity. It saves metrics and best/last
checkpoints under `saves/joint/`. Physical area/delay regression is future stage-2
work. [The preprocessing guide](docs/preprocessing.md) describes the graph
contract, input limits, stable dataset splits, caching and encoder reuse.

## Tests

See [training performance](docs/performance.md) for BF16 execution, optional
per-graph attention, and the reproducible RTX 5090 benchmark.

```bash
uv run python -m unittest discover -s tests -v
```
