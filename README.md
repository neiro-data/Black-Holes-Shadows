# Black Holes Shadows

## Resume

This repository ray-traces null geodesics to compute **black-hole shadow** images and
**gravitational-lensing** images. It covers two independent spacetimes.

- A **Weyl-coordinate Schwarzschild black hole** within a thin, massive Morgan-Morgan accretion disk.
  The physics core is `bh_shadows.core.weyl_core`.
- A **Kerr black hole** in Boyer-Lindquist `(r, theta)` coordinates. `bh_shadows.core.kerr_core`
  holds that path, and it is self-contained.

This work was presented in [this paper](https://arxiv.org/pdf/1912.08833).

Details:

- **Stack:** Python 3.11 or later, managed by [`uv`](https://docs.astral.sh/uv/). numpy, scipy,
  matplotlib, numba. ruff for lint and format, mypy for types (`uv run mypy src`).
- **What it does:** the Weyl family tabulates a lambda lookup matrix once, traces rays against
  it, and plots the result. The Kerr tracer needs no matrix. The numerics use an RKF45 (adaptive step Runge-Kutta-Fehlberg)
  integrator under numba, with `prange` in the parallel tracer.
- **Entry points:** `uv run bh-shadow <config.toml>` traces and plots one case. `uv run bh-matrix`
  builds a lambda table (the runner also builds it when it is missing).
- **Run outputs:** each run dir holds one `outputs.npz` with the keys `status`, `coord1`, `coord2` and `phi` (the traced escape azimuth, for Kerr and Weyl; the observer sits at phi = 0). Load it with `np.load`. There are no `.txt` outputs.
- **GPU:** optional MLX backend for Kerr and Weyl (`--device gpu`); see "GPU backend" below and
  [docs/GPU.md](docs/GPU.md).
- **Matrix build on GPU:** not supported. The quadrature path uses scalar adaptive `scipy.quad`, which does not port to MLX. The analytic path (MD == 0) takes milliseconds. The GPU path only reads the matrix.
- **Host requirements:** `uv` only. numba compiles on the first run of a script.

## Quick start

Run a whole case from one config (Schwarzschild ± disk, Kerr without disk):

```bash
uv sync                                                    # install
uv run bh-shadow configs/schwarzschild_nodisk.toml --n 64  # trace, save, plot (seconds)
scripts/run_cases.sh                                       # every configs/*.toml at its own n (slow)
```

Output goes to `runs/<YYYYMMDD_HHMMSS>_<name>/`, with a `run.json` that records the parameters.
A progress bar shows on stderr. Use `--no-progress` to hide it. The Weyl tracer with a disk is slow (about 8 minutes at n = 64),
and its first run also builds the lambda table: `uv run bh-shadow configs/schwarzschild_disk.toml --n 64`.

The Kerr tracer needs no matrix: `uv run bh-shadow configs/kerr_a0.9_nodisk.toml --n 64`.

A full run is expensive. Shrink the grids first and check that the code runs. See
[scripts-guide.md — Smoke test](docs/scripts-guide.md#smoke-test).

For every script, its defaults, the colormaps and the checks, see
**[scripts-guide.md](docs/scripts-guide.md)**.

## Modes

The config selects the spacetime through `[spacetime]` (`M`, `a`, `MD`, `b`). `M + MD` must be 1.0.

| Mode | Set | Tracer |
| --- | --- | --- |
| Black hole only | `MD = 0.0`, `a = 0` | Weyl (analytic lambda table). |
| Black hole and disk | `MD > 0.0`, `a = 0` | Weyl (quadrature lambda table, built once and cached). |
| Kerr | `a > 0`, `MD = 0` | Kerr (no table). |

`[observer] off_plane = true` (default `false`) moves the Weyl observer slightly above the disk plane
(`z = OBSERVER_Z_OFFSET` in `shadow_config.py`) so the disk is not seen edge-on. It needs the Weyl
backend (with or without a disk); Kerr raises `ValueError`. The equator symmetry
is lost, so the run traces twice as many rays. Example: `configs/schwarzschild_disk_MD0.6_off_plane.toml`.

## Plotting

The runner colours the image after the trace. This step uses the saved arrays and adds no
tracing time. Set the keys in the `[plot]` table of the config. All keys are in
[SCHEMA.md](docs/SCHEMA.md#config-keys--plot).

**Background** (`background`):

- `palette` (default): colours each ray by its outcome. Select the palette with `palette`. There
  are five presets, from `black_white` to `quadrant`. `quadrant` gives one colour to each quadrant
  of the lensed disk. The code is `weyl_plot.get_colormap(...)`.
- `black_hole`: white sky and a black shadow.
- `two_color`: red sky above the equator, blue sky below it.
- `four_color`: one colour for each sky quadrant (upper left red, upper right green, lower left
  yellow, lower right blue).
- `greyscale`: grey sky and a white disk.
- `image`: puts an equirectangular picture on the sky. Set its path with `image`. The path is
  relative to the config file.

**Grid** (`grid = true`):

- Draws lines on the celestial sphere. You cannot use it with `palette` or `black_hole`.
- `grid_mode = "theta"` (default): lines of constant theta and phi.
- `grid_mode = "flat_angle"` (Weyl only): horizontal and vertical lines of equal width. The lines
  are straight without a black hole. The black hole bends them.
- Weyl traces phi like Kerr. An old Weyl `outputs.npz` without `phi` uses a flat-sky phi from the
  pixel column.

**Other options:**

- `light_source = true`: adds a white glow behind the black hole. The lens makes the glow a bright
  ring. Use it only with `two_color` or `four_color`.
- `shadow_edge = true` (default `false`): draws the analytic shadow edge as a dashed cyan curve.
  For `a = 0` it is the Schwarzschild circle. For `a != 0` it is the Kerr curve. It draws nothing
  for a Weyl disk (`MD > 0`).

**Examples:**

Lensed photo of the sky (Kerr, `a = 0.9`):

```toml
[plot]
background = "image"
image = "examples/pillarsofcreation_composite.jpg"
```

Four-colour sky with a grid and a light ring:

```toml
[plot]
background = "four_color"
grid = true
grid_step_deg = 10.0
light_source = true
```

Weyl disk with straight-line grid (Weyl only):

```toml
[plot]
background = "four_color"
grid = true
grid_mode = "flat_angle"
```

Kerr shadow with its analytic edge:

```toml
[plot]
background = "black_hole"
shadow_edge = true
```

Run the ready-made examples:

```bash
uv run bh-shadow configs/kerr_a0.9_nodisk_image.toml --n 512
uv run bh-shadow configs/examples/visual_four_color_grid.toml --n 48
uv run bh-shadow configs/examples/visual_image.toml --n 48
uv run bh-shadow configs/schwarzschild_disk_MD0.6_grid.toml --n 256
```

More examples are in `configs/examples/`. For the colormaps, see
[scripts-guide.md — Colormaps](docs/scripts-guide.md#colormaps).

## Validation

`tests/test_physics_validation.py` traces small grids (n <= 64) through
`scripts/validate_physics.py` and checks the shadow against theory. Each trace runs in its own
process. Measured on CPU, observer r0 = 15, theta0 = 90 deg, h = 0.588 rad. Pitch p = 2h/(n-1).

| Check | Setup | Measured | Pass criterion |
|---|---|---|---|
| V2 Schwarzschild edge | n=64, p=0.0187; Weyl and Kerr a=0 | 0.3360 rad on all four sides (theory 0.3283; error 0.4 p) | within 1 p |
| V3 shadow pixels vs MD | n=32, n_grid=200 for MD>0; (M, MD) = (1, 0), (0.95, 0.05), (0.9, 0.1), (0.7, 0.3) | 240, 216, 208, 140; no status-3 pixels | monotone; first step <= 20% (is 10%) |
| V3 disk class | same | disk pixels 0, 288, 260, 288; 0 with `disk_visible=false` | only when the disk is visible |
| V4 spin shift | n=64, Kerr a=0.9 vs a=0 | centroid +0.126 rad along -beta (6.8 p); alpha edges +-0.2987 | shift > 1 p |
| V4 flattening | same | half-widths 0.168 (left) and 0.429 (right) | differ by >= 1 p |
| V4 mirror | raw `kerr_core` a=-0.9 vs a=0.9 flipped along beta | 0 mismatched pixels | <= 1% |
| Overlay | n=64, a=0.9, 128 boundary pixels | 100% within 1.5 p (70% within 1 p) | >= 95% within 1.5 p |

Notes:

- With MD = 0, a coarse `n_grid` (200 or 400) makes the Weyl tracer take minutes or more at n=32 to
  64. Use the default `n_grid` there; it takes about 5 s.
- `ShadowConfig` rejects `a < 0`, so the mirror check calls `kerr_core.f_paral` directly
  (`--raw-kerr-a`).

## GPU backend (optional, Apple Silicon)

On Apple Silicon, an MLX Metal kernel (float32) can trace Kerr and Weyl (Schwarzschild) rays
instead of the numba `prange` path. Install the `gpu` extra, then add `--device gpu` to the run
command:

```bash
uv sync --extra gpu
uv run bh-shadow configs/kerr_a0.9_nodisk.toml --device gpu
uv run bh-shadow configs/schwarzschild_disk.toml --n 64 --device gpu
```

- Set it with `[tracer] device = "auto|cpu|gpu"` or `--device`. `auto` resolves to `cpu` for now (`AUTO_PREFERS_GPU` stays `False`).
- `gpu` without MLX or an Apple GPU logs a warning and runs on the CPU.
- Rays that fail in float32 (status 3) are re-traced on the CPU. `run.json` records `device`
  and GPU timing keys.
- Rows per kernel launch: `GPU_CHUNK_ROWS = 32` in `bh_shadows/runner/run_shadow.py` (both backends).

On Apple Silicon the GPU trace is 23-43x faster than the CPU (n = 64, warm).
After the lambda table is built (Weyl only; Kerr needs no table), `--device gpu` makes
1024 x 1024 and even 2048 x 2048 images fast to render on an Apple Silicon MacBook.
Benchmarks, chunk sizing, CPU/GPU agreement and the `run.json` keys: see [docs/GPU.md](docs/GPU.md).

## Data files

The lambda table is a compressed NumPy file (`matrices/lambda_*.npz`) that spans `rho` in [0, 40]
and `z` in [-40, 40], at 1200 by 1200 by default. Each run writes `runs/<id>/outputs.npz`.

Full file reference in **[SCHEMA.md](docs/SCHEMA.md)**.

## For AI agents

If you are an AI agent, also read [README_AGENT.md](docs/README_AGENT.md).

## Repository Layout

```
src/bh_shadows/           the installed package
  core/                   physics: weyl_core, kerr_core, matrix (lambda table), shadow_edge
  gpu/                    optional MLX GPU backend: backend, weyl_gpu, kerr_gpu
  plots/                  weyl_plot (palettes, classification), visuals (image colouring)
  runner/                 run_shadow (bh-shadow), shadow_config (TOML config)
scripts/                  gpu_bench.py, validate_physics.py, flat_grid_preview.py, run_cases.sh
tests/                    pytest suite (grids of n <= 64)
configs/                  TOML configs; configs/examples/ for the visuals
docs/
  scripts-guide.md        every command, the colormaps, the smoke test, the checks
  SCHEMA.md               the lambda matrix, the outputs, the [plot] config keys
  GPU.md                  the GPU port: kernels, headers, differences, limitations
  README_AGENT.md         the load-bearing choices of the physics core
  initial_prompt.txt      the prompt that produced this version of the repository
README.md                 this file - what it computes and the first run
```
