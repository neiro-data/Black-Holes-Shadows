# README_AGENT.md

Load-bearing details and known mismatches. A person who only wants to produce an image does not
need this file. Referenced from [README.md](README.md).

## Read the core docstring first

Read the module docstring of `weyl_core.py` before you change the physics. It documents the
choices that the results depend on.

- The `np.abs` guard inside `xi2`.
- The `m` mode selector of `nu` and the potentials that use it.

## The two cores

| Module | Scope |
| --- | --- |
| `weyl_core.py` | The shared Weyl core: the metric, RKF45, `func` and `f_paral`, `load_matrix`. |
| `kerr_core.py` | The Kerr `(r, theta)` tracer core. It is self-contained. |
| `weyl_plot.py` | The colormaps and the shadow, disk and quadrant classification helpers. |

`weyl_core.py` holds a jitted function and a non-jitted `_i` copy for several quantities: `d1_i`,
`d2_i`, `xi2_i`, `nuD_i`, `nu_i`, `gpp_i`. `solve_observer_rho` uses the `_i` copies, because it
calls `fsolve`, and `fsolve` cannot enter a jitted function.

## Numerics

The integrator is RKF45. numba compiles it with `@jit(nopython=True)`. The production tracer,
`test_parallel_SHADOW.py`, adds `parallel=True` and `prange` for multicore execution.

The first run of any script pays the numba compile cost. A short run therefore looks slower than
it is.

## Matrix name against mode

**This mismatch is present in the tree today.**

`generate_matriz.py` writes `Mat_constA_Mbh_0.9`, with `M = 0.9`, `MD = 0.1` and `b = 3.0`.

The tracers read other names, with another `b`.

| Tracer | Reads | `b` |
| --- | --- | --- |
| `test_parallel_SHADOW.py` | `Mat_nu_disk0.1` | `6.0` |
| `test_symmetry_lensing.py` | `Mat_nu_disk0.1` | `6.0` |
| `test_Z_SHADOW.py` | `Mat_nu_disk0.0` | `6.0` |

Before a real run, align four things across `generate_matriz.py` and the tracer that you use: the
name that the generator writes, the name that the tracer reads, `MD`, and `b`. Otherwise
`load_matrix` finds no matching file.

## Plot visuals

`visuals.py` colours the final image from the arrays the tracers already return; it never traces.
Weyl `coord*` keep their sentinels (capture `[0.001, 0.001]`, disk `z + 50`) and have no phi, so
Weyl phi is a flat-sky approximation from the pixel column. See SCHEMA.md `[plot]`.

## Relation to the other repository

This repository covers the same domain as `Black-Holes-Shadows` in the same workspace. This one is
the consolidated version: shared physics cores instead of per-script duplication, a two-stage
pipeline, named colormap presets, and a documented smoke test.

The two repositories are separate. They share no history and no code.

`initial_prompt.txt` at the root records the prompt that produced this version.

## Docs map

| File | Content |
| --- | --- |
| [README.md](README.md) | The entry point: what the scripts compute, and the first run. |
| [scripts-guide.md](scripts-guide.md) | Every script, the colormaps, the smoke test, the checks. |
| [SCHEMA.md](SCHEMA.md) | The lambda matrix and the ray-tracing matrices. |
| README_AGENT.md | This file: the load-bearing choices and the name mismatch. |
