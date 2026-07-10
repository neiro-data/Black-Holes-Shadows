# Black-Holes-Shadows

Python research scripts that ray-trace null geodesics to compute
black-hole shadow and gravitational-lensing images, in two independent
spacetimes: a Weyl-coordinate Schwarzschild black hole embedded in a
Morgan-Morgan accretion disk, and a Kerr (rotating) black hole in
Boyer-Lindquist-like coordinates. Each script began life as a Jupyter
notebook and was exported to a standalone `.py` file (see "Notes on
provenance" below).

## Setup

This is a `uv`-managed Python 3.14 project (numpy, scipy, matplotlib,
numba).

```bash
uv sync
uv run python <script>.py
```

Run any script directly with `uv run python <script>.py` from this
directory.

## Pipeline

### 1. Weyl-coordinate Schwarzschild BH + Morgan-Morgan disk

This family shares a precomputed lambda-potential lookup table, since
lambda(rho, z) only has a closed form on the symmetry axis and must
otherwise be found by numerically integrating its derivatives from a
reference point at "infinity" (rho=z=40) — too slow to redo on every
photon step. It also shares a common physics core (metric potentials,
metric components, RKF45 integrator) consolidated into **`weyl_core.py`**
— see "Notes on provenance" below.

1. **`generate_matriz.py`** — the pipeline's *base* script (must run
   first): tabulates lambda on a (z, rho) grid via `lamb_Mat` (its one
   function not shared via `weyl_core.py`) and saves it to disk (e.g.
   `Mat_constA_Mbh_0.9`), consumed by the ray tracers below via
   `weyl_core.load_matrix("Mat_nu_disk*")` and bilinear interpolation.
2. One of the ray tracers integrates null geodesics over a grid of
   emission angles and produces `Mat`/`Mz` (and `Mphi`) matrices via
   `np.savetxt`:
   - **`test_Z_SHADOW.py`** — serial (plain `range`), classifies each
     pixel as BH-captured / beyond-disk / neither; its `np.savetxt` calls
     are commented out (exploratory run, shows an inline plot instead of
     saving).
   - **`test_parallel_SHADOW.py`** — the canonical/production version,
     parallelized with numba (`@jit(parallel=True)` + `prange`) for true
     multi-core execution; saves `Mat`/`Mz`.
   - **`test_symmetry_lensing.py`** — same family, focused on producing a
     lensing map rather than only a shadow classification.
3. A plotting script loads the saved matrices and produces a figure:
   **`symmetry.py`**, **`simetria_shadow.py`**, **`simetria_shadow_v2.py`**,
   **`lensing_image.py`**. `symmetry.py` in particular reconstructs the
   full image by mirroring a single ray-traced quadrant across both axes,
   exploiting the z -> -z and angular sign symmetry of the setup.

### 2. Kerr-metric family

**`lensing_r_theta.py`** is fully self-contained: it works directly with
the Kerr (r, theta) metric and its Christoffel symbols (no shared
lambda-matrix, no `np.loadtxt` dependency), integrating the full
second-order geodesic equations directly. It builds the metric and
Christoffels, integrates one photon trajectory with an adaptive RKF45
stepper, loops over a grid of emission angles to fill shadow/lensing
matrices, and saves them with `np.savetxt` for downstream plotting.

## Notes on provenance

These scripts were originally exported from Jupyter notebooks rather than
written as standalone modules. The notebook cell markers (`# In[NN]:`) and
large blocks of commented-out legacy code (earlier classification
attempts, dead lensing/diagnostic code, alternate physics formulations)
that came out of that export have since been removed to leave only the
live, executable behaviour of each script; the removed history remains
recoverable via git if it's ever needed for reference.

The four Weyl-family scripts (`generate_matriz.py`, `test_Z_SHADOW.py`,
`test_parallel_SHADOW.py`, `test_symmetry_lensing.py`) originally
duplicated a large physics-helper core near-verbatim (`simps`,
`derivative`, `d1`, `d2`, `xi2`, `nuD`, `nu`, `lambSch`,
`gpp`/`gtt`/`grr`/`gzz`, `zeta`, `dthe`/`dr`/`Pphi`/`Pt`/`dphi`/`dt`, the
`_i` observer-frame variants, `derNU`, `dlamb`, `dlamb2`, `lamb`,
`run_kut4_mod`) across files, historically tagged `#Duplicated` in the
source. That core now lives in **`weyl_core.py`**, imported by each
script via `from weyl_core import *`; see its module docstring for the
one intentional behavioural change made during extraction (`xi2`'s
`np.abs` guard) and the decorator-unification notes. Each script keeps
its own `geo` (geodesic RHS) and `func` (single-ray tracer) locally —
these differ meaningfully between scripts (shadow vs. lensing output,
disk-crossing classification, serial vs. parallel) and were deliberately
not merged; see `claude_interaction_steps.md` (Interactions 3 and 4) for
the reasoning.

## HPC

`my_job.sbatch` is a SLURM batch script for submitting these ray-tracing
runs to an HPC cluster.
