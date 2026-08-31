# CubeK Convolution Performance Plan

> Active research plan. Machine and checkout observations are dated historical
> evidence, not a description of a reproducible repository state.

Status: benchmark and generic selector optimization implemented; upstream preparation pending

## Decision

Move convolution performance work into the `bungogood/cubek` fork. CubeK becomes the primary
development repository and benchmark loop. Etive remains the end-to-end acceptance workload, and
Burn remains the integration layer that selects the CubeK implementation.

Do not add cuDNN to Burn, CubeCL, or CubeK as part of this work. Use cuDNN only as an external
performance reference.

This follows the upstream direction:

- Burn does not currently depend on cuDNN in order to preserve portability:
  [Burn discussion #2825](https://github.com/tracel-ai/burn/discussions/2825#discussioncomment-12240080).
- CubeCL aims to avoid prebuilt operation libraries where portable kernels can close the gap:
  [CubeCL PR #1440](https://github.com/tracel-ai/cubecl/pull/1440#issuecomment-5092929569).
- CubeK is being established as the canonical home for convolution implementations and benchmarks:
  [CubeK issue #164](https://github.com/tracel-ai/cubek/issues/164).

## Problem

The Etive weekend network uses eight performance-critical residual convolutions with this shape:

```text
input:          [1024, 64, 8, 8]
weight:         [64, 64, 3, 3]
output:         [1024, 64, 8, 8]
dtype:          FP16
stride:         [1, 1]
padding:        [1, 1]
dilation:       [1, 1]
groups:         1
physical data:  NHWC input/output, OHWI weight
implicit GEMM:  M=65536, N=64, K=576
GPU:            NVIDIA RTX 3070, SM86
```

Measured steady kernel performance:

| Implementation | Time per residual convolution |
| --- | ---: |
| cuDNN SM86 NHWC implicit GEMM | about 139 us |
| CubeK `simple_sync_mma` baseline | about 399 us |
| CubeK N-aware inferred selector | about 192 us |

The selector now caps the N partition to the problem width and balances LHS/RHS stage-loading work
from instruction dimensions and vector widths. It derives the validated `partition_n=8, stage_m=4`
geometry without matching the Etive shape or GPU model. The current comparison uses bias-free CubeK
measurements because a separate bias-staging correctness defect still needs repair.

The eight residual convolutions account for roughly 1.7 ms of the 2.2 ms difference between Burn
and compiled PyTorch. Existing CubeK candidates do not close the gap:

| CubeK candidate | Result |
| --- | --- |
| `SimpleSyncStrided` / MMA | Current winner |
| `SimpleAsyncStrided` / MMA | Slower, about 548 us during autotuning |
| CMMA variants | Slower |
| Specialized async cyclic | Illegal memory access |
| Specialized async strided | Illegal memory access |
| TMA variants | Unsupported on SM86 |

Layout conversion is not the primary cause. Burn already preserves channels-last trunk activations,
and pre-layouting weights improved full-network throughput by only about one percent.

## Repository Ownership

### CubeK fork

Owns:

- The isolated convolution benchmark problem.
- Strategy-by-strategy correctness tests.
- Reproduction and repair of specialized-kernel memory faults.
- Tile selection, loading, pipelining, and convolution-coordinate optimizations.
- Performance results against the current CubeK baseline and the external cuDNN reference.

Repository: <https://github.com/bungogood/cubek>

### Burn

Owns:

- Registering corrected CubeK strategies as convolution autotune candidates.
- Selecting candidates by shape, dtype, layout, and hardware capability.
- Preserving the portable fallback.

Burn should not contain the optimized convolution implementation itself.

### Etive

Owns:

- The matched 4x64 architecture benchmark.
- End-to-end self-play validation.
- Pinning a reviewed CubeK fork revision through Cargo.

Etive should not become the kernel development harness.

## Development Workflow

Use a normal development clone or worktree of `bungogood/cubek`, not Cargo's checkout under
`~/.cargo/git/checkouts`.

```bash
gh repo clone bungogood/cubek
git remote add upstream https://github.com/tracel-ai/cubek.git
git fetch upstream
git switch -c perf/etive-3x3-convolution upstream/main
```

Keep the branch based on current upstream `main`. The existing fork commit
`2744b9eb499d63b670727d0a1bd3d9694d7f22f7` contains the pitched-stride autotune-key fix, but the
convolution work should first verify whether that fix or equivalent code is already upstream.

### Current `venera` checkout

The development checkout is at `~/projects/cubek` with:

```text
branch:   perf/etive-3x3-convolution
origin:   https://github.com/bungogood/cubek.git
upstream: https://github.com/tracel-ai/cubek.git
base:     2041c03c
```

Four pre-existing uncommitted files add an experimental `SimpleSyncStridedMultiRows` strategy. They
were preserved when the detached checkout was attached to the branch. At setup time, the branch was
14 commits behind `upstream/main`.

Before Phase 1, review those changes and either save them as an explicit experiment commit or remove
them with approval. Then rebase the clean branch onto current `upstream/main`. Do not mix the
multi-row experiment, benchmark addition, specialized-kernel correctness fix, and performance work
in one commit.

The inner loop runs entirely in CubeK:

```bash
cargo test -p cubek-convolution --features cubecl/cuda,benchmarks
cargo bench -p benchmarks --bench conv2d --features cubecl/cuda
```

Only update Etive's CubeK revision after a CubeK milestone is committed and pushed. This avoids
rebuilding Burn and Etive for every kernel edit while keeping Etive builds reproducible.

Etive already patches the CubeK crates to the fork in `Cargo.toml`. At each milestone, update all
CubeK patch entries to the same new commit and regenerate `Cargo.lock`.

## Measurement Protocol

Every reported result must include:

- CubeK, CubeCL, and Burn commit IDs where applicable.
- GPU model and compute capability.
- Input, weight, output, and physical layouts.
- Dtype and accumulator dtype.
- Selected strategy and inferred blueprint.
- Warmup and measured sample counts.
- Median kernel time, not compilation or autotuning time.
- Correctness result and maximum observed error.

Benchmark both of these cases:

1. Pre-laid-out NHWC/OHWI operands, which isolates the convolution kernel.
2. Ordinary Burn NCHW/OIHW operands, which includes required layout correction.

Use a fixed environment and no concurrent GPU workloads. Warm and compile candidates before the
timed region. Nsight or sanitizer overhead must not be used as the headline timing.

## Success Criteria

### Correctness gate

- No CUDA illegal-address, misaligned-access, race, or initialization errors.
- Pass CubeK's convolution CPU-reference checks for the target problem.
- Pass adjacent boundary cases, including smaller batches and non-multiple tile dimensions.
- Preserve correctness for both pre-laid-out and ordinary strided operands.
- Keep the current working strategy available as a fallback until the new path has broad coverage.

### Performance gates

| Stage | Target median kernel time | Meaning |
| --- | ---: | --- |
| Baseline | about 350 us | Current CubeK winner |
| First useful milestone | below 250 us | At least 1.4x faster |
| Strong milestone | at or below 180 us | Within about 30% of cuDNN |
| Stretch | at or below 155 us | Within about 12% of cuDNN |

The first useful milestone is required before changing Etive's pinned revision. The strong milestone
is the target for proposing the strategy as a default Burn autotune candidate.

## Implementation Phases

### Phase 1: Establish the CubeK benchmark

Add an `etive_residual_3x3` problem to
`crates/cubek-convolution/src/eval/benchmarks/problem.rs` and expose it through the existing
`benchmarks/benches/conv2d.rs` runner.

Expand the strategy catalogue beyond its current single CMMA strategy. Include every strategy that
is valid on SM86, and report setup failures separately from runtime correctness failures.

Deliverables:

- Reproducible target problem in the CubeK benchmark catalogue.
- Correctness test for each candidate.
- Baseline timing table on the RTX 3070.
- Generated CUDA and inferred blueprint for the winning strategy.

### Phase 2: Reproduce specialized-kernel faults

Create the smallest direct test that launches `SpecializedAsyncCyclic` and
`SpecializedAsyncStrided` for the target shape. Run each strategy independently so one failure does
not invalidate the CUDA context for the rest of the suite.

Use Compute Sanitizer for diagnosis:

```bash
compute-sanitizer --target-processes all --tool memcheck \
  cargo test -p cubek-convolution --release --features cubecl/cuda,extended \
  etive_residual_3x3 -- --exact --nocapture
```

Check these likely fault sources:

- Padding predicates in the im2col input view.
- Async-copy source and destination bounds.
- Vectorized tail reads and alignment assumptions.
- Pitched NHWC and OHWI stride handling.
- Shared-memory tile size and stage offsets.
- Output tile predicates when `N=64`.
- Assumptions that differ between CMMA and MMA plane geometry.

Deliverables:

- A regression test that fails before the correction.
- A root-cause explanation tied to the invalid address.
- Passing sanitizer and CPU-reference runs after the correction.

### Phase 3: Benchmark corrected specialized paths

Compare corrected specialized candidates against `simple_sync_mma`. Keep layout correction outside
the isolated kernel timing, then measure it separately.

Reject a candidate if it is only faster because the benchmark omits work that Burn must perform.
Register it with Burn only after its input contract is represented accurately in the autotune key.

### Phase 4: Optimize the portable pipeline

Use cuDNN's selected SM86 kernel as a design clue, not as a dependency. Its relevant characteristics
are a 128x64x32 tile, four pipeline stages, Tensor Core MMA, and NHWC execution.

Investigate:

- Multi-stage asynchronous global-to-shared loading.
- Double buffering when four stages create excessive register or shared-memory pressure.
- Fewer divisions and modulo operations in the im2col coordinate mapping.
- Compile-time specialization for 3x3, stride 1, dilation 1, and padding 1.
- Shared-memory padding and swizzling.
- Vector sizes for contiguous channels.
- Persistent reuse of weight tiles across output rows.
- A blueprint specialized for large `M`, small `N`, and moderate `K`.

Prefer CubeCL feature detection and comptime specialization with a portable fallback. Avoid an
SM86-only public algorithm when the same structure can work across CUDA, ROCm, Metal, or Vulkan.

### Phase 5: Integrate through Burn autotuning

After the CubeK strategy is correct and faster:

- Add it to Burn's convolution candidate set.
- Ensure unsupported hardware returns a setup failure rather than launching an invalid kernel.
- Include physical stride alignment in the autotune key.
- Warm autotuning before CUDA graph capture.
- Confirm that channels-last output remains channels-last through GroupNorm and residual operations.

### Phase 6: Validate in Etive

Update every CubeK patch entry in Etive to the same fork commit. Then run:

```bash
cargo check --features cuda
cargo run --release --no-default-features --features cuda -- \
  bench experiments/weekend-benchmark.toml
cargo run --release --manifest-path benchmarks/architecture-throughput/Cargo.toml -- \
  --batch 1024 --warmup 50 --iterations 500 --norm group
```

Compare:

- Raw forward milliseconds per batch.
- Positions per second.
- Residual convolution kernel time and count.
- Full-batch and small-tail self-play throughput.
- Policy and value output error against the previous implementation.

### Phase 7: Upstream in reviewable pieces

Prefer separate pull requests:

1. Add the benchmark problem and strategy coverage.
2. Fix the specialized-kernel correctness fault.
3. Add the performance improvement and measurements.
4. Register the candidate in Burn after the CubeK change is accepted or stable.

Reference CubeK issue #164 and include the cuDNN number only as an external baseline.

## Non-Goals

- Adding cuDNN or cuBLAS as a required dependency.
- Creating a new Burn backend for one operation.
- Optimizing Etive-specific game or MCTS code during the kernel phase.
- Changing the network architecture to avoid the convolution problem.
- Claiming broad convolution improvement from one shape without adjacent-shape tests.
- Removing safe fallback strategies before the specialized path is proven.

## Risks

- CubeK's tile and convolution APIs are evolving quickly, so work based on the old Cargo checkout may
  conflict with current `main`.
- A fixed target shape can encourage over-specialization. Test nearby batches, channels, and spatial
  sizes before making the strategy generally eligible.
- The specialized faults may expose a deeper async-layout issue shared with matmul routines.
- RTX 3070 profiling counters may remain unavailable without elevated NVIDIA permissions. Kernel
  timing, generated CUDA inspection, and Compute Sanitizer are still available.
- Fat-LTO Burn rebuilds are slow. Keep them outside the CubeK inner loop.

## Reconsideration Gate

Revisit an optional vendor-library bridge only if all of these are true:

- The specialized paths are correct.
- The target problem is represented in upstream benchmarks.
- Current CubeK `main` and reasonable tile/pipeline experiments remain above 250 us.
- The maintainers agree that a temporary backend capability is appropriate.

Until that gate is reached, the work remains focused on portable CubeK convolution performance.
