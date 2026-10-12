# README_AGENT.md

Load-bearing details of the physics core. A person who only wants to produce an image does not
need this file. Referenced from [README.md](../README.md).

## Read the core docstring first

Read the module docstring of `src/bh_shadows/core/weyl_core.py` before you change the physics. It documents the
choices that the results depend on.

- The `np.abs` guard inside `xi2`.
- The `m` mode selector of `nu` and the potentials that use it.

## Package layout

| Module (`src/bh_shadows/`) | Scope |
| --- | --- |
| `core/weyl_core.py` | The shared Weyl core: the metric, RKF45, `func` and `f_paral`, `load_matrix`. |
| `core/kerr_core.py` | The Kerr `(r, theta)` tracer core. It is self-contained. |
| `core/matrix.py` | Builds and caches the lambda table (`bh-matrix`). |
| `core/shadow_edge.py` | Analytic shadow-edge curves. |
| `gpu/` | Optional MLX GPU backend. See [GPU.md](GPU.md). |
| `plots/weyl_plot.py` | The colormaps and the shadow, disk and quadrant classification helpers. |
| `plots/visuals.py` | Image colouring from the traced arrays. |
| `runner/` | `bh-shadow`: config loading, tracing, saving. |

`weyl_core.py` holds a jitted function and a non-jitted `_i` copy for several quantities: `d1_i`,
`d2_i`, `xi2_i`, `nuD_i`, `nu_i`, `gpp_i`. `solve_observer_rho` uses the `_i` copies, because it
calls `fsolve`, and `fsolve` cannot enter a jitted function.

## Numerics

The integrator is RKF45. numba compiles it with `@jit(nopython=True)`. `f_paral` and `retrace_points`
add `parallel=True` and `prange` for multicore execution.

The first run of any script pays the numba compile cost. A short run therefore looks slower than
it is.

## Plot visuals

`plots/visuals.py` colours the final image from the arrays the tracers already return; it never traces.
Weyl `coord*` keep their sentinels (capture `[0.001, 0.001]`, disk `z + 50`). Weyl traces `phi`
on CPU and GPU, like Kerr. Only an old Weyl `outputs.npz` without `phi` falls back to a flat-sky phi
from the pixel column. See SCHEMA.md `[plot]`.

## Relation to the other repository

This repository covers the same domain as `Black-Holes-Shadows` in the same workspace. This one is
the consolidated version: shared physics cores instead of per-script duplication, a two-stage
pipeline, named colormap presets, and a documented smoke test.

The two repositories are separate. They share no history and no code.

`docs/initial_prompt.txt` records the prompt that produced this version.

## Docs map

| File | Content |
| --- | --- |
| [README.md](../README.md) | The entry point: what the scripts compute, and the first run. |
| [scripts-guide.md](scripts-guide.md) | Every script, the colormaps, the smoke test, the checks. |
| [SCHEMA.md](SCHEMA.md) | The lambda matrix, the outputs and the `[plot]` config keys. |
| [GPU.md](GPU.md) | The GPU port: kernels, headers, differences, limitations. |
| README_AGENT.md | This file: the load-bearing choices. |
