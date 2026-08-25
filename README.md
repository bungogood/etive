# Etive

Etive is an AlphaZero-style game-learning project written in Rust with
[Candle](https://github.com/huggingface/candle) as its tensor backend. The
current foundation includes a tested, allocation-free Othello rules engine,
tic-tac-toe with an exact minimax oracle, and direct-to-batch neural state
encoding. Synchronous PUCT search uses contiguous node and edge arenas with
subtree reuse and automatic compaction after each played move.

The project is intentionally kept in one crate while its game, search, and
inference boundaries are established through working code.

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

## Candle Backends

CPU is the default. Select Accelerate, CUDA, cuDNN, or Metal with the matching
Cargo feature. CUDA and cuDNN require the corresponding NVIDIA development
libraries.

Use `cargo run -- --help` to display the available commands.

## Engine Protocol

Etive can run as a [Go Text Protocol v2](https://www.gnu.org/software/gnugo/gnugo_19.html)
Othello engine for integration with match runners and engines such as Edax and
Egaroucid:

```bash
cargo run --release -- gtp
cargo run --release --features cudnn -- gtp --checkpoint checkpoints/model.safetensors --batch-size 128
```

Without a checkpoint the protocol player selects the first legal move for rules
testing. With a checkpoint it uses persistent, leaf-parallel MCTS and batches up
to `--batch-size` positions in each network invocation. Diagnostics must go to
stderr while GTP is active because stdout is reserved for protocol responses.

## Evaluation

Compare two checkpoints with color-balanced games and reproducible openings:

```bash
cargo run --release --features cudnn -- eval previous.safetensors contender.safetensors
```

## Training

Training runs are defined entirely by TOML files. The included configuration
starts a fresh 24-hour residual-network experiment:

```bash
cargo run --release --features cudnn -- train experiments/residual-10x128-24h.toml
```

Every completed generation saves model weights, AdamW state, replay data,
metrics, and elapsed run state. The same command resumes automatically when its
configured output directory contains an Etive run.

Use `--clean` to discard a recognized run and start again. Etive refuses to
clean an output directory without its run metadata. Relative checkpoint and
output paths are resolved from the configuration file's directory. A run copies
its configuration into the output directory; changed configurations are
archived when a run resumes.

### Tic-Tac-Toe Validation

Tic-tac-toe is a small deterministic fixture, not a training target. Its tests
cover all 5,478 reachable positions, exact minimax outcomes, policy action
mapping, MCTS selection and backup, terminal inference bypass, and agreement
between synchronous and batched search. Othello will supply the actual
self-play and learning pipeline.

## Development

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --release
cargo test --release -- --ignored
cargo bench --bench rules
cargo bench --bench search
```

Etive is licensed under the [MIT License](LICENSE).
