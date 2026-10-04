# Black-Holes-Shadows-one-shot_effort-medium--low

## Resume

This repository ray-traces null geodesics to compute **black-hole shadow** images and
**gravitational-lensing** images. It covers two independent spacetimes.

- A **Weyl-coordinate Schwarzschild black hole** in a thin, massive Morgan-Morgan accretion disk.
  The physics core is `bh_shadows.core.weyl_core`.
- A **Kerr black hole** in Boyer-Lindquist `(r, theta)` coordinates. `bh_shadows.core.kerr_core`
  holds that path, and it is self-contained.

Details:

- **Stack:** Python 3.11 or later, managed by [`uv`](https://docs.astral.sh/uv/). numpy, scipy,
  matplotlib, numba. ruff for lint and format.
- **What it does:** the Weyl family tabulates a lambda lookup matrix once, traces rays against
  it, and plots the result. The Kerr tracer needs no matrix. The numerics use an RKF45
  integrator under numba, with `prange` in the parallel tracer.
- **GPU:** optional MLX backend for Kerr and Weyl (`--device gpu`); see "GPU backend" below and
  [docs/GPU.md](docs/GPU.md).
- **Entry points:** `uv run bh-shadow <config.toml>` traces and plots one case. `uv run bh-matrix`
  builds a lambda table (the runner also builds it when it is missing).
- **Run outputs:** each run dir holds one `outputs.npz` with the keys `status`, `coord1`, `coord2` (and `phi` for Kerr). Load it with `np.load`. There are no `.txt` outputs.
- **Matrix build on GPU:** not supported. The quadrature path uses scalar adaptive `scipy.quad`, which does not port to MLX. The analytic path (MD == 0) takes milliseconds. The GPU path only reads the matrix.
- **Host requirements:** `uv` only. numba compiles on the first run of a script.

## GPU backend (optional, Apple Silicon, Kerr and Weyl)

An MLX Metal kernel (float32) can trace Kerr and Weyl (Schwarzschild) rays instead of the numba `prange` path.

```bash
uv sync --extra gpu
uv run bh-shadow configs/kerr_a0.9_nodisk.toml --device gpu
uv run bh-shadow configs/schwarzschild_disk.toml --n 64 --device gpu
```

- Set it with `[tracer] device = "auto|cpu|gpu"` or `--device`. `auto` resolves to `cpu` for now (`AUTO_PREFERS_GPU` stays `False`).
- `gpu` without MLX or an Apple GPU logs a warning and runs on the CPU.
- Rays that fail in float32 (status 3) are re-traced on the CPU. `run.json` records `device`
  and `gpu_cpu_fallback_pixels`, plus `gpu_chunk_rows`, `gpu_dispatch_s_max`,
  `gpu_dispatch_s_mean`, `matrix_s` (matrix build + load) and `retrace_s`.
- Rows per kernel launch: `GPU_CHUNK_ROWS = 32` in `bh_shadows/runner/run_shadow.py` (both backends).
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
If you are an AI agent, read also [README_AGENT.md](docs/README_AGENT.md).

## Quick start

Run a whole case from one config (Schwarzschild ± disk, Kerr without disk):

```bash
uv sync                                                  # install
uv run bh-shadow configs/schwarzschild_disk.toml --n 64  # trace, save, plot
scripts/run_cases.sh                                     # all shipped configs, one process each
```

Output goes to `runs/<YYYYMMDD_HHMMSS>_<name>/`, with a `run.json` that records the parameters.
A progress bar shows on stderr. Use `--no-progress` to hide it. The Weyl tracer is slow (about 8 minutes at n = 64).

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

## Plotting

The `palette` background picks a palette by name, with `weyl_plot.get_colormap(...)`. Five
presets exist, from `black_white` to `quadrant`, which gives one colour per lensed-disk quadrant.

The config runner can also colour the final image from the ray outcome, with `[plot] background`
(`palette`, `black_hole`, `two_color`, `greyscale`, `four_color`, `image`) and an optional
celestial-sphere `grid`. This is a post-process on the traced arrays and adds no tracing cost. The
Weyl backend has no traced phi, so left/right and grid phi lines use a flat-sky approximation. Keys
`light_source = true` adds a white glow behind the black hole (two_color/four_color), which lenses into a bright ring. Keys
are in [SCHEMA.md](docs/SCHEMA.md#config-keys--plot); examples are in `configs/examples/`.

`[plot] shadow_edge = true` (default `false`) draws the analytic shadow edge as a dashed cyan
curve: the Schwarzschild circle for `a = 0`, the Kerr curve for `a != 0`. It draws nothing for a Weyl
disk (`MD > 0`).

See [scripts-guide.md — Colormaps](docs/scripts-guide.md#colormaps).

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

## Data files

The lambda table is a compressed NumPy file (`matrices/lambda_*.npz`) that spans `rho` in [0, 40]
and `z` in [-40, 40], at 1200 by 1200 by default. Each run writes `runs/<id>/outputs.npz`.

Full file reference in **[SCHEMA.md](docs/SCHEMA.md)**.

## Layout

```
src/bh_shadows/           the installed package
  core/                   physics: weyl_core, kerr_core, matrix (lambda table), shadow_edge
  gpu/                    optional MLX GPU backend: backend, weyl_gpu, kerr_gpu
  plots/                  weyl_plot (palettes, classification), visuals (image colouring)
  runner/                 run_shadow (bh-shadow), shadow_config (TOML config)
scripts/                  gpu_bench.py, validate_physics.py, run_cases.sh
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
