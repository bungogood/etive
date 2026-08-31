# Etive

Etive is an AlphaZero-style Othello learning project written in Rust with
[Burn](https://github.com/tracel-ai/burn) and CubeCL. It includes a tested,
allocation-free rules engine, direct-to-batch neural state encoding, residual
policy/value networks, and leaf-parallel PUCT search. Search uses contiguous
node and edge arenas, batched inference, subtree reuse, and automatic
compaction after each played move. A tic-tac-toe minimax oracle is retained as
a test fixture for the generic search implementation.

## Othello Rules

The Othello board uses two side-relative `u64` bitboards. Legal destinations
and flip masks are generated across all eight directions without allocation.
Move application provides checked paths for external input and explicit
unchecked paths for search code that has already established legality.

Bit numbering runs from `A1 = 0` through `H8 = 63`. Positions use eight
slash-separated rows followed by the side to move:

```text
......../......../......../...BW.../...WB.../......../......../........ b
```

Perft treats a mandatory pass as one ply and returns one leaf when neither
player can move, matching [Aart Bik's published
counts](https://www.aartbik.com/strategy.php#reversi):

| Depth | Nodes |
| ---: | ---: |
| 8 | 390,216 |
| 9 | 3,005,288 |
| 10 | 24,571,284 |
| 11 | 212,258,800 |
| 12 | 1,939,886,636 |

Run an opening-position perft with:

```bash
cargo run --release -- perft 11
```

## Burn Backends

Burn Flex CPU is the default for fast local development. Use
`--no-default-features --features cpu` for CubeCL CPU or
`--no-default-features --features cuda` for CubeCL CUDA. Apple platforms can
use `--no-default-features --features metal`. CUDA requires the corresponding
NVIDIA development libraries.

CUDA inference uses FP16 and training uses FP32 with CubeCL kernel autotuning.

Use `cargo run -- --help` to display all commands. The main workflows are:

| Command | Purpose |
| --- | --- |
| `gtp` | Run an Othello engine over Go Text Protocol v2 |
| `eval` | Compare two checkpoints with fixed-search games |
| `train` | Start or resume a TOML-configured training run |
| `bench` | Measure the configured self-play pipeline |
| `diagnose-replay` | Compare checkpoint predictions with replay targets |
| `train-frozen` | Run controlled optimization over frozen replay shards |
| `perft` | Verify opening-position move-generation counts |

## Engine Protocol

Etive can run as a [Go Text Protocol v2](https://www.gnu.org/software/gnugo/gnugo_19.html)
Othello engine for integration with match runners and engines such as Edax and
Egaroucid:

```bash
cargo run --release -- gtp
cargo run --release --no-default-features --features cuda -- gtp --checkpoint checkpoints/model.burnpack --batch-size 128
```

Without a checkpoint the protocol player selects the first legal move for rules
testing. With a checkpoint it uses persistent, leaf-parallel MCTS and batches up
to `--batch-size` positions in each network invocation. Diagnostics must go to
stderr while GTP is active because stdout is reserved for protocol responses.

## Evaluation

Compare two checkpoints with color-balanced games and reproducible openings:

```bash
cargo run --release --no-default-features --features cuda -- eval previous.burnpack contender.burnpack
```

## Training

Training runs are defined entirely by TOML files. Start with the smoke profile
when validating a new environment:

```bash
cargo run --release --no-default-features --features cuda -- train experiments/smoke.toml --clean
```

The current weekend candidate uses a 4-block, 64-channel network, 4,096
self-play games per generation across eight workers, 256 simulations per move,
and champion gating:

```bash
cargo test --release --no-default-features --features cuda othello::training::tests::fixed_batch_overfit_gate -- --ignored --exact
CUBECL_ENVIRONMENT=etive-weekend-v1 cargo run --release --no-default-features --features cuda -- train experiments/weekend-4x64-256.toml --clean
```

Run the ignored overfit test before starting a new production run. It requires
the policy head to fit a fixed soft target and the value head to fit a fixed
outcome. Weekend candidates are evaluated against the current champion on a
fixed color-paired opening suite. Rejected candidates do not replace the
self-play network; the learner and optimizer continue from their latest state.

Every completed generation saves model weights, AdamW state, model architecture,
replay data, metrics, champion generation, and elapsed run state. The same
command resumes automatically when its configured output directory contains an
Etive run. Metrics include policy target entropy and policy KL in addition to
raw cross-entropy, so target fit can be distinguished from target sharpness.

Measure the real self-play worker, batching, and inference pipeline:

```bash
CUBECL_ENVIRONMENT=etive-bench-v1 RUST_LOG=info cargo run --release --no-default-features --features cuda -- bench experiments/weekend-benchmark.toml
```

The benchmark loads model and actor settings from the experiment TOML and runs
without training or writing checkpoints.

Controlled regression commands are available for validated replay data:

```bash
cargo run --release -- diagnose-replay checkpoint.burnpack replay.bin --rows 20000
cargo run --release -- train-frozen checkpoint.burnpack optimizer.burnpack --replay replay.bin --steps 1000 --output artifacts/frozen-run
```

Model and optimizer checkpoints use Burn's burnpack format. Checkpoints from
the former Candle implementation are not compatible and must not be used to
resume a Burn run.

Training logs are written to stderr. Set `RUST_LOG` to adjust their verbosity:

```bash
RUST_LOG=debug cargo run --release --no-default-features --features cuda -- train experiments/weekend-4x64-256.toml
```

Use `--clean` to discard a recognized run and start again. Etive refuses to
clean an output directory without its run metadata. Relative checkpoint and
output paths are resolved from the configuration file's directory. A run copies
its configuration into the output directory; changed configurations are
archived when a run resumes.

## Development

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --release
cargo test --release published_deep_initial_position_perft -- --ignored --exact
```

## Documentation And Research

- [`experiments/README.md`](experiments/README.md) identifies current and historical run profiles.
- [`benchmarks/README.md`](benchmarks/README.md) indexes reproducible benchmarks and investigation data.
- [`docs/README.md`](docs/README.md) distinguishes active plans from historical debugging records.
- [`notebooks/README.md`](notebooks/README.md) explains the locked metrics-analysis environment.

Etive is licensed under the [MIT License](LICENSE).
