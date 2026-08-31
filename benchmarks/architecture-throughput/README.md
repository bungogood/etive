# Architecture Throughput Benchmark

Matched raw-forward benchmark for Etive's weekend Othello network in Burn and PyTorch.

The benchmark uses:

- FP16 CUDA inference with NCHW inputs.
- Batch size 1,024 by default.
- A 2-to-64 channel 3x3 stem followed by GroupNorm and ReLU.
- Four 64-channel residual blocks, each containing two 3x3 convolutions and GroupNorm.
- The same policy and value heads as Etive.
- Explicit GPU synchronization before and after the timed region.
- No input upload, output readback, MCTS, or self-play work in the measurement.

## Burn

```bash
CUBECL_ENVIRONMENT=etive-architecture-bench \
  cargo run --release --manifest-path benchmarks/architecture-throughput/Cargo.toml -- \
  --batch 1024 --warmup 50 --iterations 500 --norm group
```

The Burn dependency is pinned to the same Burn and CubeK revisions as Etive. The historical Burn
results below also used experimental GroupNorm changes stored only in the measurement machine's
local Burn Cargo checkout. They are retained as investigation data but are not reproducible from
this directory alone; a clean machine benchmarks stock Burn.

## PyTorch

```bash
python benchmarks/architecture-throughput/pytorch.py \
  --batch 1024 --warmup 50 --iterations 500 --mode both --norm group
```

`--mode both` reports eager PyTorch and `torch.compile(..., mode="reduce-overhead")` separately.
Compilation and warmup happen outside the timed region.

Use `--norm group`, `--norm batch`, or `--norm none` to compare the current architecture with
inference BatchNorm or an upper bound that removes trunk normalization entirely.

The Burn benchmark also accepts `--weight-layout channels-last` to pre-layout convolution weights
as OHWI storage while retaining their logical OIHW shape. The default is `contiguous` OIHW storage.
Run with `--validate-group-norm` to compare CUDA GroupNorm against an FP64 CPU reference for both
NCHW and channels-last physical layouts.

Use the same GPU, batch, warmup, and iteration count for both commands. The headline metric is
`positions_per_second`; `milliseconds_per_batch` is useful for profiler comparisons.

## RTX 3070 Results

Measured on `venera` with batch 1,024, 50 warmup iterations, and 500 measured iterations:

| Backend | Version | ms/batch | positions/s |
| --- | --- | ---: | ---: |
| Burn with GroupNorm epilogue fusion | local patched checkout | 4.155 | 246,470 |
| PyTorch eager | 2.9.0+cu128 | 5.028 | 203,650 |
| PyTorch compiled | 2.9.0+cu128 | 1.779 | 575,558 |
| PyTorch eager | 2.13.0+cu130 | 5.054 | 202,613 |
| PyTorch compiled | 2.13.0+cu130 | 1.803 | 567,810 |

Burn is about 21% faster than eager PyTorch here, but only 42.8% of PyTorch 2.9 compiled
throughput. `torch.compile` uses `fullgraph=True` and `mode="reduce-overhead"`; its compilation and
warmup costs are excluded.

### Normalization and layout variants

| Backend | Normalization | Weight layout | ms/batch | positions/s |
| --- | --- | --- | ---: | ---: |
| Burn, fused Welford GroupNorm | GroupNorm | OIHW | 4.171 | 245,504 |
| Burn | BatchNorm | OIHW | 3.550 | 288,490 |
| Burn | None | OIHW | 3.460 | 295,933 |
| Burn, fused Welford GroupNorm | GroupNorm | OHWI | 4.129 | 248,017 |
| Burn | None | OHWI | 3.432 | 298,334 |
| Burn, fused two-pass GroupNorm | GroupNorm | OHWI | 4.015 | 255,059 |
| PyTorch compiled | GroupNorm | generated channels-last | 1.779 | 575,558 |
| PyTorch compiled | BatchNorm | generated channels-last | 1.593 | 642,624 |
| PyTorch compiled | None | generated channels-last | 1.595 | 641,948 |

The two-pass GroupNorm row includes an additional experimental local-checkout change. It computes
the mean and squared deviations in separate FP32 reductions, then runs the fused affine, residual,
and ReLU epilogue. Against an FP64 CPU reference, its maximum absolute FP16 error was 0.0004884 for
both NCHW and channels-last storage.

Removing normalization leaves a 1.84 ms/batch gap between Burn and compiled PyTorch, so convolution
is the primary difference. Nsight measured PyTorch's eight steady residual cuDNN NHWC convolutions
at about 139 us each. Burn's autotuner selected `simple_sync_mma`; its available alternatives were
slower, and its residual convolution kernels measured about 350 us each. Pre-layouting weights only
removes a roughly 1% reorder cost because Burn already preserves channels-last trunk activations.
