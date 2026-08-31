# Training Regression Investigation

Status: completed 2026-08-31.

The investigation isolated an initialization-sensitive collapse at a `1e-3`
learning rate and validated a corrected `3e-4` 4x64 preflight. Start with
[`report.md`](report.md) for conclusions and [`artifacts.md`](artifacts.md) for
historical provenance.

| File | Contents |
| --- | --- |
| `corrected-4x64-256-4h.csv` | Complete corrected 4x64 series through generation 68 |
| `trial-8x64-256-8h.csv` | Preserved 8x64 comparison snapshot |
| `arena.csv` | Fixed-checkpoint match results |
| `diagnostics.csv` | Frozen replay diagnostics |

Checkpoint cells use run-relative identifiers such as
`trial-4x64-256-4h/generation-0001.burnpack`. The checkpoint binaries and
the original 781 MB remote artifact bundle are not distributed here, so the
CSV data supports analysis but does not make the historical run independently
reproducible.
