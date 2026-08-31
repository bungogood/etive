# Training Regression Report

## Status

This historical investigation was completed on 2026-08-31. The failed 72-hour candidate was preserved remotely, the regression was reproduced, and the primary failure was isolated to initialization-sensitive optimization at the original learning rate. The remote bundle is not published with this repository, so the report is evidentiary rather than independently reproducible.

## Principal Finding

The generation-1 learner seed deterministically changes whether the network survives the first updates:

| Seed | Learning rate | Steps | Policy KL | Legal mass | Value correlation | Value standard deviation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.001 | 1,000 | 0.944 | 65.9% | 0.337 | 0.292 |
| 8 | 0.000998 | 1,000 | 2.321 | 13.5% | 0.000 | approximately 0 |
| 8 | 0.0003 | 1,000 | 0.739 | 73.7% | 0.367 | 0.348 |
| 8 | 0.0003 | 5,000 | 0.262 | 95.2% | 0.416 | 0.369 |

The failed experiment reseeded generation-1 training with seed 8. At approximately `1e-3`, this sampling/symmetry sequence drove the network to position-independent outputs. Lowering the learning rate to `3e-4` prevented the collapse with the identical checkpoint, optimizer, replay, seed, backend, and batch size.

The stable 5,000-step checkpoint beat generation 0 by 398-2 over 400 paired games at 256 simulations.

## Backend Check

The first collapse-producing update was reproduced on CUDA and local Flex:

| Backend | Policy loss | Target entropy | Policy KL | Value loss |
| --- | ---: | ---: | ---: | ---: |
| CUDA | 4.292100 | 1.830230 | 2.461871 | 0.983232 |
| Flex | 4.292063 | 1.830230 | 2.461833 | 0.983180 |

The agreement rules out a CUDA-specific optimizer or convolution failure as the leading cause.

## Checkpoint Output Collapse

Across 20,000 frozen replay positions, generations 4, 8, and 12 produced effectively constant value predictions and nearly position-independent policies. FP16 and FP32 diagnostics agreed, so inference casting did not cause the collapse.

Raw data:

- `benchmarks/training-regression/arena.csv`
- `benchmarks/training-regression/diagnostics.csv`
- `benchmarks/training-regression/artifacts.md`

## Corrective Direction

1. Reduce initial learning rate from `1e-3` to `3e-4` and final learning rate proportionally to `3e-5`.
2. Add learner/champion separation so a regressed learner cannot generate self-play data.
3. Continue the learner and optimizer after failed evaluation instead of restoring them.
4. Run a two-to-four-hour preflight with diagnostics against fixed anchors before another 72-hour run.

## Corrected Preflight

The four-hour preflight uses `experiments/trial-4x64-256-4h.toml`. Its first evaluation passed:

| Generation | Train policy KL | Validation policy KL | Train value loss | Validation value loss | Arena result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.475 | 0.260 | 0.797 | 0.754 | Not evaluated |
| 2 | 0.219 | 0.205 | 0.773 | 0.761 | Not evaluated |
| 3 | 0.192 | 0.183 | 0.763 | 0.771 | Not evaluated |
| 4 | 0.177 | 0.173 | 0.757 | 0.760 | 400-0 against generation 0 |

Generation 4 was the first promoted self-play champion. The completed preflight reached generation 68 with champion generation 48. Training and validation losses tracked closely, policy KL declined steadily, and value loss remained materially below the approximately 1.0 constant-prediction baseline. The complete series is stored in `corrected-4x64-256-4h.csv`.
