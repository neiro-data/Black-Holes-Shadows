# Black-Holes-Shadows

Python research scripts that ray-trace null geodesics to compute
black-hole shadow and gravitational-lensing images, in two independent
spacetimes: a Weyl-coordinate Schwarzschild black hole embedded in a
Morgan-Morgan accretion disk, and a Kerr (rotating) black hole in
Boyer-Lindquist-like coordinates. Each script began life as a Jupyter
notebook and was exported to a standalone `.py` file, so cell markers and
legacy exploratory code remain in place (see "Notes on provenance" below).

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
photon step.

1. **`generate_matriz.py`** — tabulates lambda on a (z, rho) grid and
   saves it to disk (e.g. `Mat_constA_Mbh_0.9`), consumed by the ray
   tracers below via `np.loadtxt("Mat_nu_disk*")` and bilinear
   interpolation.
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

These scripts are exported from Jupyter notebooks (`# In[NN]:` cell
markers throughout) rather than written as standalone modules, so large
blocks of commented-out legacy code (earlier classification attempts,
dead lensing/diagnostic code, alternate physics formulations) are
preserved for reference. Because the original notebooks were cloned
rather than refactored into a shared module, several helper functions
(e.g. `simps`, `derivative`, `d1`, `d2`, `xi2`, `nuD`, `nu`, `lambSch`,
`gpp`/`gtt`/`grr`/`gzz`, the `_i` observer-frame variants, `derNU`,
`dlamb`, `dlamb2`) are duplicated near-verbatim across files and tagged
`#Duplicated` in the source.

## HPC

`my_job.sbatch` is a SLURM batch script for submitting these ray-tracing
runs to an HPC cluster.
