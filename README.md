# Black-Holes-Shadows-one-shot_effort-medium--low

## Resume

This repository ray-traces null geodesics to compute **black-hole shadow** images and
**gravitational-lensing** images. It covers two independent spacetimes.

- A **Weyl-coordinate Schwarzschild black hole** in a thin, massive Morgan-Morgan accretion disk.
  Most scripts use the shared physics core, `weyl_core.py`.
- A **Kerr black hole** in Boyer-Lindquist `(r, theta)` coordinates. `kerr_core.py` holds that
  path, and it is self-contained.

Details:

- **Stack:** Python 3.11 or later, managed by [`uv`](https://docs.astral.sh/uv/). numpy, scipy,
  matplotlib, numba. ruff for lint and format.
- **What it does:** the Weyl family tabulates a lambda lookup matrix once, traces rays against
  it, and plots the result. The Kerr driver integrates on its own. The numerics use an RKF45
  integrator under numba, with `prange` in the production tracer.
- **GPU:** optional MLX backend for Kerr and Weyl (`--device gpu`); see "GPU backend" below.
- **Entry points:** `generate_matriz.py` for the Weyl family, which must run first.
  `lensing_r_theta.py` for the Kerr family.
- **Run outputs:** each run dir holds one `outputs.npz` with the keys `status`, `coord1`, `coord2` (and `phi` for Kerr). Load it with `np.load`. There are no `.txt` outputs.
- **Matrix build on GPU:** not supported. The quadrature path uses scalar adaptive `scipy.quad`, which does not port to MLX. The analytic path (MD == 0) takes milliseconds. The GPU path only reads the matrix.
- **Host requirements:** `uv` only. numba compiles on the first run of a script.

## GPU backend (optional, Apple Silicon, Kerr and Weyl)

An MLX Metal kernel (float32) can trace Kerr and Weyl (Schwarzschild) rays instead of the numba `prange` path.

```bash
uv sync --extra gpu
uv run python run_shadow.py configs/kerr_a0.9_nodisk.toml --device gpu
uv run python run_shadow.py configs/schwarzschild_disk.toml --n 64 --device gpu
```

- Set it with `[tracer] device = "auto|cpu|gpu"` or `--device`. `auto` resolves to `cpu` for now (`AUTO_PREFERS_GPU` stays `False`).
- `gpu` without MLX or an Apple GPU logs a warning and runs on the CPU.
- Rays that fail in float32 (status 3) are re-traced on the CPU. `run.json` records `device`
  and `gpu_cpu_fallback_pixels`, plus `gpu_chunk_rows`, `gpu_dispatch_s_max`,
  `gpu_dispatch_s_mean`, `matrix_s` (matrix build + load) and `retrace_s`.
- Rows per kernel launch: `GPU_CHUNK_ROWS = 32` in `run_shadow.py` (both backends).
- Benchmark one config (n <= 64; one config per process): `uv run python scripts/gpu_bench.py
  --config configs/schwarzschild_disk.toml --n 64 --n-grid 1200 --device gpu --chunk 32`.
  `--agree` compares CPU and GPU on the same table.

**Measured (Apple Silicon, 2026-10-04, n = 64, warm = median of 3, matrix cached):**

| Config | CPU warm | GPU warm (chunk 32) | Speedup | Dispatch max (chunk 32) |
| --- | --- | --- | --- | --- |
| `schwarzschild_disk` (n_grid 1200) | 1.82 s | 0.079 s | 23x | 0.032 s |
| `schwarzschild_nodisk` | 1.90 s | 0.063 s | 30x | 0.019 s |
| `kerr_a0.9_nodisk` | 0.81 s | 0.019 s | 43x | 0.019 s |

Cold start (first trace, includes numba/Metal compile): CPU 4-6 s, GPU 0.2-0.9 s. `matrix_s` is
0.05 s on a cached table; the quadrature build is separate and scales as `n_grid^2`.
No float32 fallback pixels in any of these runs. All speedups are above 1.5x, so
`AUTO_PREFERS_GPU` can be flipped in a later PR (not changed here).

**Chunk rule** (`T_BUDGET` = 1.0 s per dispatch; an assumption, the macOS watchdog limit is not
documented). Weyl at n = 64, disk: max dispatch 0.031 / 0.032 / 0.031 / 0.032 s for chunk
4 / 8 / 16 / 32, so it is launch-bound. Per-thread cost `c = 0.0318 / (32 x 32)` = 3.1e-5 s
(pessimistic), projected for n = 1024: `t(32) = c x 32 x 512` = 0.51 s <= 1 s. Kerr: real
n = 4096 run, 0.19 s mean per 16-row dispatch; max/mean ratio at n = 64 is q = 1.04, so
`t(32) = 2 x 0.19 x 1.04` = 0.40 s <= 1 s. Both backends get 32.

**Weyl CPU vs GPU agreement on the real quadrature table** (n = 64, n_grid = 1200, M = 0.9,
MD = 0.1): status mismatch 0 of 4096 pixels; median |d coord1| on escaped pixels 5.3e-4,
|d coord2| 6.3e-5; 1140 disk pixels on both devices, median |d coord1| 6.3e-4; 0 fallback pixels.

## Agent read
If you are an AI agent, read also [README_AGENT.md](README_AGENT.md).

## Quick start

```bash
uv sync                                  # install
uv run python generate_matriz.py         # stage 1 - tabulate the lookup matrix
uv run python test_parallel_SHADOW.py    # stage 2 - trace the shadow, multicore
uv run python symmetry.py                # stage 3 - plot
```

Or run a whole case from one config (Schwarzschild ± disk, Kerr without disk):

```bash
uv run python run_shadow.py configs/schwarzschild_disk.toml --n 64
scripts/run_cases.sh                     # all shipped configs, one process each
```

Output goes to `runs/<YYYYMMDD_HHMMSS>_<name>/`, with a `run.json` that records the parameters.
A progress bar shows on stderr. Use `--no-progress` to hide it. The Weyl tracer is slow (about 8 minutes at n = 64).

The Kerr driver needs no matrix.

```bash
uv run python lensing_r_theta.py
```

A full run is expensive. Shrink the grids first and check that the code runs. See
[scripts-guide.md — Smoke test](scripts-guide.md#smoke-test).

For every script, its defaults, the colormaps and the checks, see
**[scripts-guide.md](scripts-guide.md)**.

## Modes

Each Weyl tracer selects its spacetime through the disk mass `MD`, at the top of `__main__`.

| Mode | Set | Load |
| --- | --- | --- |
| Black hole only | `MD = 0.0` | A matrix tabulated with `MD = 0.0`. |
| Black hole and disk | `MD > 0.0` | The matching matrix, for example `M = 0.9, MD = 0.1`. `M + MD` must be 1.0. |

The name of the loaded matrix must match the chosen `MD`. Change `MD` and the argument of
`wc.load_matrix(...)` together.

The names in the tree do not match today. Read
[README_AGENT.md — Matrix name against mode](README_AGENT.md#matrix-name-against-mode) before a
real run.

## Plotting

A plotting script picks a palette by name, with `weyl_plot.get_colormap(...)`. Five presets
exist, from `black_white` to `quadrant`, which gives one colour per lensed-disk quadrant.

An optional hook, `weyl_plot.load_grid_from_image(path)`, builds a grid mask from an image that
you supply.

The config runner can also colour the final image from the ray outcome, with `[plot] background`
(`palette`, `black_hole`, `two_color`, `greyscale`, `four_color`, `image`) and an optional
celestial-sphere `grid`. This is a post-process on the traced arrays and adds no tracing cost. The
Weyl backend has no traced phi, so left/right and grid phi lines use a flat-sky approximation. Keys
`light_source = true` adds a white glow behind the black hole (two_color/four_color), which lenses into a bright ring. Keys
are in [SCHEMA.md](SCHEMA.md#config-keys--plot); examples are in `configs/examples/`.

See [scripts-guide.md — Colormaps](scripts-guide.md#colormaps).

## Data files

Every intermediate file is a plain-text matrix. The lambda matrix spans `rho` in [0, 40] and `z`
in [-40, 40], at 1200 by 1200. The tracers write one quadrant of the emission-angle grid, and the
plotting stage mirrors it.

Full file reference in **[SCHEMA.md](SCHEMA.md)**.

## Layout

```
weyl_core.py              the shared Weyl core: metric, RKF45, func/f_paral, load_matrix
kerr_core.py              the self-contained Kerr (r, theta) tracer core
weyl_plot.py              the colormaps and the shadow/disk/quadrant classification helpers
generate_matriz.py        stage 1 - tabulates the lambda lookup matrix; run this first
test_parallel_SHADOW.py   stage 2 - the production shadow tracer, numba prange
test_Z_SHADOW.py          stage 2 - the serial shadow tracer, with live plotting
test_symmetry_lensing.py  stage 2 - the lensing tracer; saves Mat, Mz, Mphi
lensing_r_theta.py        the Kerr driver; needs no lookup matrix
symmetry.py               stage 3 - plots from the saved matrices
simetria_shadow.py        stage 3 - plots from the saved matrices
simetria_shadow_v2.py     stage 3 - plots from the saved matrices
lensing_image.py          stage 3 - plots from the saved matrices
initial_prompt.txt        the prompt that produced this version of the repository
README.md                 this file - what it computes and the first run
scripts-guide.md          every script, the colormaps, the smoke test, the checks
SCHEMA.md                 the lambda matrix and the ray-tracing matrices
README_AGENT.md           the load-bearing choices and the name mismatch
```
