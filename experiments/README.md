# Experiments

Experiment paths are resolved relative to their TOML file. Training output is
written beneath the ignored `checkpoints/` directory.

| Configuration | Status | Purpose |
| --- | --- | --- |
| `smoke.toml` | Current | Short end-to-end environment validation |
| `trial-4x64-256-4h.toml` | Completed baseline | Corrected 4x64 preflight used by the regression report |
| `trial-8x64-256-8h.toml` | Completed comparison | Paired-LOS 8x64 comparison trial |
| `weekend-4x64-256.toml` | Current candidate | 72-hour 4x64 run after the corrected preflight |
| `weekend-benchmark.toml` | Current utility | Self-play throughput benchmark; training fields are schema-only |

Run `cargo run --release -- train <config>` to resume an existing recognized
run. Add `--clean` only when intentionally discarding that configuration's
checkpoint directory. Run configuration is immutable after initialization;
changed experiments require a new output path or `--clean`.
