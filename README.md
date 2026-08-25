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

CPU is the default backend. The `candle` command runs a small tensor operation
to verify the selected device:

```bash
cargo run -- candle
cargo run --release --features cuda -- candle
cargo run --release --features cudnn -- candle
cargo run --release --features metal -- candle
```

CUDA and cuDNN builds require the corresponding NVIDIA development libraries;
enabling a Cargo feature does not install them.

Use `cargo run -- --help` to display the available commands.

## Engine Protocol

Etive can run as a [Go Text Protocol v2](https://www.gnu.org/software/gnugo/gnugo_19.html)
Othello engine for integration with match runners and engines such as Edax and
Egaroucid:

```bash
cargo run --release -- gtp
cargo run --release --features cudnn -- gtp --checkpoint checkpoints/model.safetensors
```

Without a checkpoint the protocol player selects the first legal move for rules
testing. With a checkpoint it uses MCTS. Diagnostics must go to stderr while
GTP is active because stdout is reserved for protocol responses.

## Random Candle Search

Run complete Othello games using PUCT search and a reproducibly initialized
residual Candle policy/value network:

```bash
cargo run --release -- mcts --simulations 128 --seed 7
cargo run --release -- mcts --games 64 --batch-size 64 --simulations 128
```

Terminal positions bypass the network. Their exact side-to-move-relative game
result is negated across each search edge during backup.

## Training

Training runs are defined entirely by TOML files. The included configuration
starts a fresh 24-hour residual-network experiment:

```bash
cargo run --release --features cudnn -- learn experiments/residual-10x128-24h.toml
```

Every completed generation atomically saves model weights, AdamW state, replay
data, metrics, and elapsed run state. Resume the same total runtime after an
interruption with:

```bash
cargo run --release --features cudnn -- learn experiments/residual-10x128-24h.toml --resume
```

Relative checkpoint and output paths are resolved from the configuration file's
directory. A run copies its configuration into the output directory. If a
resume changes that configuration, the previous version is archived beside the
run before the new settings take effect.

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
