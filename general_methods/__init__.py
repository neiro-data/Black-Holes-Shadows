#!/usr/bin/env python
# coding: utf-8
"""Shared Weyl-coordinate physics core for the Schwarzschild BH + Morgan-Morgan
disk ray tracers in this repository (generate_matriz.py, test_parallel_SHADOW.py,
test_Z_SHADOW.py, test_symmetry_lensing.py).

This package holds the helpers that were previously copy-pasted near-verbatim
across those four scripts (tagged `#Duplicated` in their history) -- numerical
utilities, the nu/lambda metric potentials, the metric components and photon
momenta/velocities, the RKF45 integrator, and the non-jitted observer-frame
variants used to locate the initial observer position via fsolve.

Formerly a single flat module (weyl_core.py); split by concern into:
- `mathematical_formulas`: simps, derivative, run_kut4_mod, load_matrix, lamb.
- `physical_quantities`: d1, d2, xi2 (and their non-jitted "_i" copies).
- `physical_potentials`: nu, nuD, lambSch and their derivatives (derNU, dlamb,
  dlamb2), plus non-jitted "_i" copies.
- `spacetime_metrics`: gtt, grr, gzz, gpp, zeta, Pphi, Pt, dthe, dr, dphi, dt
  as free functions (required for numba-jit call compatibility -- see that
  module's docstring), plus `Metric`/`MetricDerivatives` convenience classes.

Each consuming script keeps its own `geo` (geodesic RHS), `func` (single-ray
tracer), driver, and I/O -- those differ meaningfully between scripts (shadow
vs lensing output, disk-crossing classification, serial vs parallel) and are
not part of this shared core.

Canonicalization notes (see claude_interaction_steps.md, Interaction 4):
- `xi2` here uses the `np.abs(...)`-guarded form (previously only in
  generate_matriz.py) rather than the unguarded form the three tracers used --
  this matches how the lookup table was actually built and avoids NaN for
  points where the radicand is a tiny negative number near the disk edge.
- All jitted helpers use plain `@jit(nopython=True)`, EXCEPT `run_kut4_mod`,
  which keeps `@jit(nopython=True, parallel=True)`. `parallel=True` on scalar
  leaf helpers (nu, lamb, gtt, ...) was a no-op in the original
  test_parallel_SHADOW.py, so those were safely unified. `run_kut4_mod` is
  different: it is a *higher-order* function that receives another jitted
  function (`geo`) as a first-class callback argument `F`. In
  test_parallel_SHADOW.py, `geo`/`func` still carry `parallel=True` (kept
  as-is, since they are that file's local variant layer); calling that
  parallel-decorated `geo` through a *plain*-nopython `run_kut4_mod`
  triggers numba's workqueue threading layer to abort with "Concurrent
  access has been detected" (confirmed by testing: this reproduces
  regardless of whether f_paral's own prange loop is active). Keeping
  `run_kut4_mod` itself `parallel=True` -- matching how it was originally
  paired with test_parallel_SHADOW.py's other parallel=True functions --
  avoids the mismatch; it remains a correctness-neutral no-op for the two
  callers (test_Z_SHADOW.py, test_symmetry_lensing.py) whose own `geo`/`func`
  are plain nopython. Real *effective* parallelism (prange over the pixel
  grid) still lives only in test_parallel_SHADOW.py's own `f_paral`.

`lamb`'s Mat_nu dependency (provisional): `lamb` bilinearly interpolates a
pre-tabulated lambda matrix held in `mathematical_formulas.Mat_nu`. Call
`load_matrix(path)` once at script startup (before any `lamb` call) to set it.
This mirrors the previous per-script `Mat_nu = np.loadtxt(...)` pattern; a
more explicit (parameter-threaded) design is a possible follow-up.
"""

from .mathematical_formulas import *  # noqa: F401,F403
from .physical_quantities import *  # noqa: F401,F403
from .physical_potentials import *  # noqa: F401,F403
from .spacetime_metrics import *  # noqa: F401,F403

from . import mathematical_formulas as mathematical_formulas
from . import physical_quantities as physical_quantities
from . import physical_potentials as physical_potentials
from . import spacetime_metrics as spacetime_metrics

__all__ = (
    mathematical_formulas.__all__
    + physical_quantities.__all__
    + physical_potentials.__all__
    + spacetime_metrics.__all__
)
