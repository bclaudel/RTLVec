# Stage-1 performance

Use BF16 autocast with the existing attention layout:

```bash
OMP_NUM_THREADS=4 uv run python train.py \
  --data-dir data/deepcircuitx/stage1-full --output-dir saves/deepcircuitx-full \
  --num_epochs 50 --max_nodes_per_batch 4096 --transformer_num_heads 4 \
  --stratified_masking --random_flip_posenc --learning_rate 2e-5 \
  --precision bf16
```

`--precision bf16` applies CUDA autocast to forward computations and losses.
Parameters and optimizer states remain FP32; backward and optimizer updates
run outside autocast. A BF16-capable CUDA GPU is required. The default remains
`--precision fp32`. BF16 changes numerical precision: compare validation loss
and accuracy over a longer run before judging quality.

`--attention-layout graph` represents circuits as independent sequences,
grouped into power-of-two size buckets and padded within each bucket. Padding
positions are blocked as attention keys and discarded from the output.
Single-graph buckets need no padding. This avoids cross-graph attention while
preserving learned parameters, node ordering and reconstruction objectives.
Node masking still occurs after the GNN; EP still encodes the complete graph.

The default remains `--attention-layout dense`. Existing checkpoint weights
work with both layouts. New checkpoints record the layout in `model_config`;
older checkpoints default to dense. With dropout disabled, FP32 outputs and
gradients match within numerical tolerance. Layout changes consume dropout
randomness differently, so training trajectories are not identical.

## RTX 5090 measurements, 2026-10-10

Median wall times over five measured steps after two warm-up steps, using the
real joint training loop with stratified masking, positional sign flips,
class-balanced focal loss, negative-edge sampling and AdamW. The model has
eight GIN layers and eight Transformer layers with four attention heads.
CUDA-event hooks separate the components and add some overhead; their intervals
include host launch gaps. Data loading, collation, checkpoint writing and
validation are outside the step timings. Each configuration also passed MNM
and EP validation on its batch.

| Batch | Nodes | Dense FP32 | Dense BF16 | Graph FP32 | Graph BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mixed sizes, 16 graphs | 3,110 | 141 ms | 80 ms | 134 ms | 120 ms |
| Near-median sizes, 16 graphs | 3,344 | 159 ms | 106 ms | 83 ms | 67 ms |
| Single maximum-size graph | 2,048 | 71 ms | 48 ms | 69 ms | 43 ms |

For mixed-size dense FP32, the two Transformer forwards together took about
30 ms and Transformer backward took 75 ms. GNN forward/backward took 6 ms and
3 ms. The Transformer is the main measured compute cost.

BF16 improved all three cases. Graph attention helped similarly sized graphs,
but launching several buckets limited its benefit on mixed batches; dense
BF16 was faster there. These are three diagnostic batches, not a full-epoch
throughput or model-quality comparison. Start with BF16 and dense attention;
evaluate graph attention against the actual batch distribution.

Reproduce the benchmark without creating training checkpoints:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 uv run python benchmark_training.py \
  --data-dir data/deepcircuitx/stage1-full --repeats 5 --warmup 2
```

The ignored local report at `data/deepcircuitx/performance/benchmark.json`
records the dataset fingerprint, model configuration, batch sizes, component
timings, GPU memory use and smoke-test metrics.
