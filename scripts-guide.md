# scripts-guide.md

Every script of this repository, grouped by stage, plus the smoke test and the checks.
Referenced from [README.md](README.md).

Run any script from the root of the repository.

```bash
uv sync
uv run python <script>.py
```

## Stage 1 — Tabulate the lambda matrix

```bash
uv run python generate_matriz.py
```

This is the base step of the Weyl family. It must run before any Weyl tracer. It tabulates the
lambda potential on a 1200 by 1200 grid and writes the grid with `np.savez_compressed`
(`matrices/lambda_*.npz`, key `lam`).

The file name and the parameters are set at the top of `__main__`. The current values are
`M = 0.9`, `MD = 0.1`, `b = 3.0`, and the output name `Mat_constA_Mbh_0.9`.

Read [README_AGENT.md — Matrix name against mode](README_AGENT.md#matrix-name-against-mode)
before a real run. The name that this script writes and the name that a tracer reads do not
match today.

## Stage 2 — Trace the rays, Weyl family

| Script | Behaviour | Defaults |
| --- | --- | --- |
| `test_parallel_SHADOW.py` | The production shadow tracer. numba `prange`, multicore. | `M = 0.9`, `b = 6.0`, loads `Mat_nu_disk0.1`. |
| `test_Z_SHADOW.py` | The serial shadow tracer, with live three-colour plotting. | `M = 1.0`, `b = 6.0`, loads `Mat_nu_disk0.0`. |
| `test_symmetry_lensing.py` | The lensing tracer. It tracks `phi` and saves `Mat`, `Mz` and `Mphi`. | `M = 0.9`, `b = 6.0`, loads `Mat_nu_disk0.1`. |

Each tracer selects its spacetime through the disk mass `MD`, at the top of `__main__`.

- **Black hole only:** set `MD = 0.0`, and load a matrix tabulated with `MD = 0.0`.
- **Black hole and disk:** set `MD > 0.0`, and load the matching matrix, for example `MD = 0.1`.

The name of the loaded matrix **must** match the chosen `MD`. Change `MD` and the argument of
`wc.load_matrix(...)` together. `load_matrix` must run before the first trace.

## Stage 3 — Plot

```bash
uv run python symmetry.py
uv run python simetria_shadow.py
uv run python simetria_shadow_v2.py
uv run python lensing_image.py
```

Each script reads the saved `Mat`, `Mz` and `Mphi` files and writes a figure.

### Colormaps

A plotting script picks a palette by name, with `weyl_plot.get_colormap(...)`.

| Name | Look |
| --- | --- |
| `black_white` | White background. Black black hole and grid. |
| `white_black_grey` | White background. Black black hole. Grey disk and ring. |
| `grey_black_white` | Grey background. Black black hole and grid. White disk. |
| `darkgrey_black` | Dark-grey background. Black black hole and grid. |
| `quadrant` | Black black hole and grid. One colour per lensed-disk quadrant. |

`weyl_plot.load_grid_from_image(path, threshold=0.5)` is an optional hook. It builds a grid mask
from an image that you supply.

## Kerr family

```bash
uv run python lensing_r_theta.py
```

This driver needs no lookup matrix. It runs on its own, against `kerr_core.py`.

## Smoke test

A full run is expensive: the `alfa` and `beta` grids are large and numba compiles first. To check
that the code runs, make the grids small. Do not run at full resolution.

1. **Check the imports and the JIT compile.** This traces nothing and it is fast.

   ```bash
   uv run python -c "import weyl_core, kerr_core, weyl_plot; print('imports OK')"
   ```

2. **Use a tiny matrix and a tiny trace.** Lower the resolution in the driver under test. In
   `generate_matriz.py`:

   ```python
   z   = np.linspace(40.0, -40.0, 20)   # the real value is 1200
   rho = np.linspace(40.0,   0.0, 20)   # the real value is 1200
   ```

   In a tracer, for example `test_Z_SHADOW.py`, trace a few rays only:

   ```python
   alfaa = np.linspace(-np.arctan(10 / 15), np.arctan(10 / 15), 10)  # the real value is 80
   betaa = np.linspace( np.arctan(10 / 15), -np.arctan(10 / 15), 10)
   ```

   A finite result grid, with no NaN value, means that the physics and the numba path are
   healthy. Restore the original resolutions for a real render.

3. **For the Kerr tracer**, lower the grid count of 752 in `lensing_r_theta.py` to about 10.

## Checks

```bash
uv run ruff format .   # format
uv run ruff check .    # lint
```

The repository has no test suite and no type checker.

## Config-driven runner

```bash
uv run python run_shadow.py configs/kerr_a0.9_nodisk.toml --n 16
```

- **Progress:** a `tqdm` bar shows rows done, elapsed time and ETA on stderr. Use `--no-progress`
  to hide it. The parallel path calls the numba tracer in row chunks, and the results are
  identical to a single call.
- **Cores:** the runner uses 80% of the system cores (at least 1) for numba. The count is stored
  in `run.json` as `numba_threads`.
- **Output folder:** `runs/<YYYYMMDD_HHMMSS>_<name>/` (local time). Each run gets its own folder.
- **Files:** `config.resolved.toml`, `run.json`, `status.txt`, `coord1.txt`, `coord2.txt`,
  `phi.txt` (Kerr), `image.png`.
- **Visuals:** set `[plot] background` (see [SCHEMA.md](SCHEMA.md#config-keys--plot)); the image
  is composed from the saved arrays by `visuals.py`. Examples:
  `uv run python run_shadow.py configs/examples/visual_two_color.toml --n 16`, and likewise
  `visual_black_hole`, `visual_greyscale`, `visual_four_color_grid`, `visual_image`.
- **`run.json`:** `run_id`, `timestamp`, `timestamp_iso`, `name`, `config_path`, `backend`, `M`,
  `a`, `MD`, `b`, `distance`, `theta0`, `n`, `half_angle`, `disk_visible`, `parallel`,
  `matrix_source`, `n_grid`, `palette`, `elapsed_s`, `shadow_pixels`, `numba_threads`,
  `versions`, `status`.
