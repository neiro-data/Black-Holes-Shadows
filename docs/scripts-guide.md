# scripts-guide.md

Every command and script of this repository, plus the smoke test and the checks.
Referenced from [README.md](../README.md).

Run all commands from the root of the repository. `uv sync` installs the `bh_shadows` package
and the commands `bh-shadow` and `bh-matrix`.

```bash
uv sync
```

## Config-driven runner

```bash
uv run bh-shadow configs/kerr_a0.9_nodisk.toml --n 16
```

- **Progress:** a `tqdm` bar shows rows done, elapsed time and ETA on stderr. Use `--no-progress`
  to hide it. The parallel path calls the numba tracer in row chunks, and the results are
  identical to a single call.
- **Cores:** the runner uses 80% of the system cores (at least 1) for numba. The count is stored
  in `run.json` as `numba_threads`.
- **Output folder:** `runs/<YYYYMMDD_HHMMSS>_<name>/` (local time). Each run gets its own folder.
- **Files:** `config.resolved.toml`, `run.json`, `outputs.npz` (keys `status`, `coord1`, `coord2`,
  `phi`), `image.png`.
- **Visuals:** set `[plot] background` (see [SCHEMA.md](SCHEMA.md#config-keys--plot)). The image
  is composed from the saved arrays by `bh_shadows.plots.visuals`. Examples:
  `uv run bh-shadow configs/examples/visual_two_color.toml --n 16`, and likewise
  `visual_black_hole`, `visual_greyscale`, `visual_four_color_grid`, `visual_image`.
- **GPU:** `--device gpu`. See [GPU.md](GPU.md).
- **`run.json`:** `run_id`, `timestamp`, `timestamp_iso`, `name`, `config_path`, `backend`, `M`,
  `a`, `MD`, `b`, `distance`, `theta0`, `n`, `half_angle`, `disk_visible`, `parallel`,
  `matrix_source`, `n_grid`, `palette`, `elapsed_s`, `shadow_pixels`, `numba_threads`,
  `versions`, `status`.

## Lambda matrix

```bash
uv run bh-matrix --M 0.9 --MD 0.1 --b 6.0 --n-grid 1200
```

The Weyl tracer needs this table. The runner builds it when it is missing, so you do not have to
run this command first. The table covers `rho` in [0, 40] and `z` in [-40, 40]. It goes to
`matrices/lambda_M*_MD*_b*_n*.npz` (key `lam`), with a JSON sidecar. `--source` is `auto`,
`analytic` (only `MD = 0`) or `quadrature`.

## Checks

```bash
uv run ruff format .     # format
uv run ruff check .      # lint
uv run python -m pytest  # tests in tests/
```

Use `uv run python -m pytest`, not `uv run pytest`. If `.venv` was copied from another clone, the
`.venv/bin/pytest` script keeps a shebang to that clone's Python, and the tests fail with
`ModuleNotFoundError: No module named 'bh_shadows'`. `python -m pytest` always uses this venv. To
repair the scripts, delete `.venv` and run `uv sync`.

The tests use grids of n of 64 or less. The repository has no type checker.

## Smoke test

A full run is expensive. To check that the code runs, use a small grid:

```bash
uv run python -c "import bh_shadows.core.weyl_core, bh_shadows.core.kerr_core; print('OK')"
uv run bh-shadow configs/kerr_a0.9_nodisk.toml --n 16 --no-progress
uv run bh-shadow configs/schwarzschild_nodisk.toml --n 16 --no-progress
```

A result with no status-3 (red) pixels means that the physics and the numba path are healthy.

## Physics validation

```bash
uv run python scripts/validate_physics.py --n 64 --a 0.9
```

The script traces one grid (n of 64 or less) and prints JSON: status counts, central-line shadow
edges, centroid and half-widths. Options: `--config`, `--M --MD --a --theta0`, `--backend`,
`--disk-visible`, `--n-grid`, `--matrices-dir`, `--raw-kerr-a` (trace `kerr_core` directly,
allows a < 0), `--save-status`. One config per process.

## GPU benchmark and agreement check

```bash
uv run python scripts/gpu_bench.py --config configs/schwarzschild_disk.toml --n 64 --n-grid 1200 --device gpu --chunk 32 --repeats 3
uv run python scripts/gpu_bench.py --config configs/schwarzschild_disk.toml --n 64 --agree
```

The script times one config per process (n of 64 or less) and prints one JSON line: cold time,
warm trace time, matrix time, dispatch max and mean, and fallback pixels. With `--agree` it traces
on the CPU and then on the GPU with the same frozen table and prints the agreement metrics.
`--matrices-dir` sets the matrix folder.

## Colormaps

`plot_status` (the `palette` background) picks a palette by name, with
`weyl_plot.get_colormap(...)`.

| Name | Look |
| --- | --- |
| `black_white` | White background. Black black hole and grid. |
| `white_black_grey` | White background. Black black hole. Grey disk and ring. |
| `grey_black_white` | Grey background. Black black hole and grid. White disk. |
| `darkgrey_black` | Dark-grey background. Black black hole and grid. |
| `quadrant` | Black black hole and grid. One colour per lensed-disk quadrant. |
