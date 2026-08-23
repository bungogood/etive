# Etive

Etive is an AlphaZero-style game-learning project written in Rust with
[Candle](https://github.com/huggingface/candle) as its tensor backend. The
current foundation is a tested, allocation-free Othello rules engine.
Tic-tac-toe will provide an exact minimax oracle for validating MCTS and the
learning loop before the system is applied to Othello.

The project is intentionally kept in one crate while its game, search,
inference, and training boundaries are established through working code.

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
```

The current protocol player selects the first legal move deterministically. It
exists to validate interoperability before MCTS supplies competitive moves.
Diagnostics must go to stderr while GTP is active because stdout is reserved
for protocol responses.

## Development

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --release
cargo test --release -- --ignored
cargo bench --bench rules
```

Etive is licensed under the [MIT License](LICENSE).
