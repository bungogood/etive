# Training Regression Artifacts

## Run

- Remote output: `/home/jonathan/projects/etive-rootcause-selector/checkpoints/weekend-4x64-256`
- Preserved size: 781 MB
- Completed replay files: 12
- Pending self-play: no
- Local Git baseline: `4ff86718f4035dba732f17a88a9d7bdd5dea7f9e`
- Remote source mirror has no Git metadata, so exact deployed source and binary hashes are recorded below.

## Deployed Source

| Artifact | SHA-256 |
| --- | --- |
| `src/othello/experiment.rs` | `5442afeb795e720510973635e3e42e641df451adcd81aef99cb7595004c70c87` |
| `src/othello/training.rs` | `34a285113244feb30187fa23321cde84242b790dff41fae408cf8706463deeb2` |
| `src/othello/actors.rs` | `7987a4aec42a0c5ab062765684672d03f9358e4e0915e7ea4c319afaa3706df3` |
| CUDA release binary | `b8a675bd5c4b622ca7db8d626a02212c3d3988744c1b9b5fb96b7943e8ddd1aa` |

These hashes identify the deployed investigation snapshot based on commit `4ff8671`; the worktree
continued evolving after deployment and is not expected to match them now. The remote `Cargo.toml`
and `Cargo.lock` intentionally differ because the remote mirror uses its local CubeK patch checkout.

## Run Metadata

| Artifact | SHA-256 |
| --- | --- |
| `config.toml` | `d445ed4ec71cc119467d46ae514e6b2b4f62d135792a80f1d83f5f1d274d7a58` |
| `state.toml` | `8dab960c265a61deaacf3a96c3b89a8db2eaa7dfb2fd50a24c572a96e8748a8e` |
| `metrics.csv` | `71b7e8fa3142e952ffeb1b413d05e9beb005659537e70439dd96fda93f85b2ca` |

## Anchor Checkpoints

| Generation | Network SHA-256 | Optimizer SHA-256 |
| ---: | --- | --- |
| 0 | `d95b28aca8955d43021d15caa9e408e6801488278329fdbca03d26f50e0ddeed` | `d51cafc1a9901c1d26d289a9e29457bb4bb041f704ee783577e890d516927813` |
| 4 | `dfef781b218c03519992d5697cd312edd39bce5a933ab5da7e57ff66ce937b61` | `cff2cf4f5409e576171bf5222b7989385e4ad41043f5bd066c4a7adf58c66bf6` |
| 8 | `eaa53b0405e6f32cd1774c065db5a786d59fbe24abddd55b29af335f797f1c83` | `90dfb32459f55187f4914999fb7c2bff5ee079ef474a82b000ae05d4fe83f866` |
| 12 | `f7427edf7a48a8f595790aa4997c21d7eb8cd5f1c6cdf3d260bbec7f470a687a` | `3b5cf6d381573323b7a2709daaf950d803802688b6a199aa1e7e5585b31b1e8b` |

Arena source data is stored in `benchmarks/training-regression-arena.csv`.
