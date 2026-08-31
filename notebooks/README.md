# Notebooks

The environment is locked with uv. From the repository root:

```bash
uv sync --project notebooks
uv run --project notebooks jupyter lab notebooks
```

`training-metrics.ipynb` reads the tracked CSV files under
`benchmarks/training-regression/`. Its `RUNS` dictionary controls which series
are compared. Keep committed notebooks free of execution counts and outputs;
generated notebook checkpoints and virtual environments are ignored.
