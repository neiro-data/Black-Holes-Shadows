"""Shared physics core for the Weyl-coordinate Schwarzschild BH + Morgan-Morgan disk family.

All Weyl-family scripts (``test_parallel_SHADOW.py``, ``test_Z_SHADOW.py``,
``test_symmetry_lensing.py``, ``symmetry.py``, ``generate_matriz.py``) import this module
via ``from weyl_core import *`` instead of redefining the physics. This file is a
deduplication of what used to be near-identical copies of the same functions pasted
across those scripts; the canonical signatures kept here match ``test_parallel_SHADOW.py``
(the production tracer), which is the only variant that keeps the ``m`` disk/BH selector.

Load-bearing conventions (do not "fix" these without re-deriving the physics):

* **The ``m`` selector.** Wherever a potential function takes an ``m`` argument:
  ``m == 0`` selects the pure Schwarzschild potential (``nuSch``, black hole only),
  ``m == 1`` selects the pure Morgan-Morgan disk potential (``nuD``), and ``m == 2``
  selects the combined BH + disk potential (``nuSch + nuD``). The geodesic/metric code
  always calls with ``m == 2``; ``m == 0``/``m == 1`` exist so a caller can isolate a
  single source (e.g. to build a BH-only lookup matrix by setting ``MD = 0``).

* **The ``xi2`` ``np.abs`` guard.** ``xi2`` takes ``np.sqrt(np.abs(...))`` of the argument
  of the outer square root. The un-guarded expression can go slightly negative due to
  floating-point cancellation near the disk plane/axis; without the guard this raises
  inside numba's ``nopython`` mode. This is intentional and must be kept.

* **``load_matrix`` / the ``Mat_nu`` global.** ``lamb`` bilinearly interpolates the
  precomputed non-linear lambda-potential lookup table into the module-level global
  ``Mat_nu`` (built by ``generate_matriz.py`` via ``lamb_Mat``). Numba freezes globals
  referenced inside a ``@jit(nopython=True)`` function at first compilation, so
  ``load_matrix(path)`` **must** be called before the first traced ray (i.e. before the
  first call into any function that transitively calls ``lamb``). The matrix is *not*
  threaded through ``geo``/``func``/``f_paral`` signatures on purpose — that would require
  passing a large read-only array through every recursive RK step, which is both slower
  and a bigger churn to the numba-compiled call graph than the existing global.

* **``parallel=True`` on all numba-jitted core functions.** This mirrors
  ``test_parallel_SHADOW.py`` (the production/canonical implementation) even for
  functions that are also called from the serial driver (``test_Z_SHADOW.py``);
  numba's ``parallel=True`` only auto-parallelizes loops it can prove are safe (e.g. the
  ``prange`` loop in ``f_paral``), so it is a strict superset behaviourally and keeps a
  single compiled core shared by both the parallel and serial drivers.
"""

import math

import numpy as np
from numba import jit, prange
from scipy.optimize import fsolve

try:
    import scipy.integrate as sci
except ImportError:  # pragma: no cover - scipy always available in this project
    sci = None

# ---------------------------------------------------------------------------
# Module-level lambda-potential lookup matrix.
#
# Initialised to a tiny placeholder so the module imports (and numba can trace types)
# even before a real matrix is loaded. Call `load_matrix(path)` BEFORE tracing any ray.
# ---------------------------------------------------------------------------
Mat_nu = np.zeros((4, 4))


def load_matrix(path: str) -> None:
    """Load the tabulated lambda-potential matrix produced by generate_matriz.py (`.npz`, key
    `lam`; any other suffix is read as legacy plain text).

    Must be called before the first ray trace: numba freezes the `Mat_nu` global the
    first time a `@jit(nopython=True)` function referencing it is compiled.
    """
    global Mat_nu
    if str(path).endswith(".npz"):
        with np.load(path) as f:
            Mat_nu = f["lam"]
    else:  # legacy plain-text tables
        Mat_nu = np.loadtxt(path)


# ---------------------------------------------------------------------------
# Numerical integration / differentiation helpers
# ---------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def simps(f, l, z, a, b, N, *args):
    """Composite Simpson's rule integrator of `f` along axis `l` (0=first arg, 1=second)."""
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = (b - a) / N
    if l == 0:
        x = np.linspace(a, b, N + 1)
        y = f(x, z, *args)
        S = dx / 3 * np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2])
    elif l == 1:
        x = np.linspace(a, b, N + 1)
        y = f(z, x, *args)
        S = dx / 3 * np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2])
    return S


@jit(nopython=True, parallel=True)
def derivative(f, l, x, y, hder, *args):
    """Central finite difference of `f` w.r.t. its first (l=0) or second (l=1) argument."""
    if l == 0:
        return 0.5 * (f(x + np.abs(hder), y, *args) - f(x - np.abs(hder), y, *args)) / np.abs(hder)
    elif l == 1:
        return 0.5 * (f(x, y + np.abs(hder), *args) - f(x, y - np.abs(hder), *args)) / np.abs(hder)
    elif l != 0 and l != 1:
        print("Error. Check 'l' number")


# ---------------------------------------------------------------------------
# Weyl-coordinate distances to the two Schwarzschild "rod" endpoints (+/-M on the axis)
# ---------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def d1(rho, z, M):
    return np.sqrt(rho**2 + (z - M) ** 2)


@jit(nopython=True, parallel=True)
def d2(rho, z, M):
    return np.sqrt(rho**2 + (z + M) ** 2)


# ---------------------------------------------------------------------------
# Inverted Morgan-Morgan disk potentials
# ---------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def xi2(R, z, a):
    """Oblate-spheroidal-like coordinate used by the Morgan-Morgan disk potential.

    Wrapped in `np.abs` (load-bearing, see module docstring): floating-point cancellation
    can push the radicand's argument slightly negative near z=0, R>=a, which would
    otherwise raise inside numba nopython mode.
    """
    return np.sqrt(
        np.abs(
            (a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2) ** 2 + 2 * (a**2 + R**2) * z**2 + z**4))
            / (R**2 + z**2)
        )
    ) / np.sqrt(2)


@jit(nopython=True, parallel=True)
def nuD(R, z, M, a):
    """Morgan-Morgan (inverted) thin-disk contribution to the nu potential."""
    if xi2(R, z, a) == 0.0:
        return (
            M
            * np.sqrt(
                np.abs(
                    a**2
                    - R**2
                    - z**2
                    + np.sqrt((a**2 - R**2) ** 2 + 2 * (a**2 + R**2) * z**2 + z**4)
                )
            )
            * (
                -3 * a**2
                + R**2
                + z**2
                + 3 * np.sqrt((a**2 - R**2) ** 2 + 2 * (a**2 + R**2) * z**2 + z**4)
            )
        ) / (np.sqrt(2) * np.pi * (R**2 + z**2) ** 2) - (
            2 * M * (-(a**2 * (R**2 - 2 * z**2)) + 2 * (R**2 + z**2) ** 2) * np.pi / 2
        ) / (np.pi * (R**2 + z**2) ** 2.5)
    else:
        return (
            M
            * np.sqrt(
                np.abs(
                    a**2
                    - R**2
                    - z**2
                    + np.sqrt((a**2 - R**2) ** 2 + 2 * (a**2 + R**2) * z**2 + z**4)
                )
            )
            * (
                -3 * a**2
                + R**2
                + z**2
                + 3 * np.sqrt((a**2 - R**2) ** 2 + 2 * (a**2 + R**2) * z**2 + z**4)
            )
        ) / (np.sqrt(2) * np.pi * (R**2 + z**2) ** 2) - (
            2
            * M
            * (-(a**2 * (R**2 - 2 * z**2)) + 2 * (R**2 + z**2) ** 2)
            * np.arctan(
                np.sqrt(2)
                * np.sqrt(
                    (R**2 + z**2)
                    / np.abs(
                        a**2
                        - R**2
                        - z**2
                        + np.sqrt((a**2 - R**2) ** 2 + 2 * (a**2 + R**2) * z**2 + z**4)
                    )
                )
            )
        ) / (np.pi * (R**2 + z**2) ** 2.5)


# l = 0 -> derivative w.r.t. RHO ; l = 1 -> derivative w.r.t. Z
# m = 0 -> nuSch (BH only) ; m = 1 -> nuD (disk only) ; m = 2 -> nuSch + nuD (BH + disk)
@jit(nopython=True, parallel=True)
def nu(rho, z, M, MD, b, m):
    """nu potential: BH (Schwarzschild), disk (Morgan-Morgan) or the sum, per the `m` selector."""
    nuSch = math.log(
        (d1(rho, z, M) + d2(rho, z, M) - 2 * M) / (d1(rho, z, M) + d2(rho, z, M) + 2 * M)
    )
    if m == 0:
        return nuSch
    elif m == 1:
        return nuD(rho, z, MD, b)
    elif m == 2:
        return nuSch + nuD(rho, z, MD, b)


@jit(nopython=True, parallel=True)
def lambSch(rho, z, M, MD, b):
    """Analytic Schwarzschild lambda potential (used as the outer boundary value for lamb_Mat)."""
    sigma = np.sqrt((rho**2 + z**2 + M**2) ** 2 - 4 * z**2 * M**2)
    return np.log(((d1(rho, z, M) + d2(rho, z, M)) ** 2 - 4 * M**2) / (4 * sigma))


# ---------------------------------------------------------------------------
# Observer-frame (`_i`) variants: plain Python (not numba), used only once to solve
# for the initial observer's rho via `scipy.optimize.fsolve`.
# ---------------------------------------------------------------------------
def d1_i(rho, z, M):
    return np.sqrt(rho**2 + (z - M) ** 2)


def d2_i(rho, z, M):
    return np.sqrt(rho**2 + (z + M) ** 2)


def xi2_i(R, z, a):
    return np.sqrt(
        (a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2) ** 2 + 2 * (a**2 + R**2) * z**2 + z**4))
        / (R**2 + z**2)
    ) / np.sqrt(2)


def nuD_i(R, z, M, a):
    if xi2_i(R, z, a) == 0.0:
        return (
            M
            * np.sqrt(
                np.abs(
                    a**2
                    - R**2
                    - z**2
                    + np.sqrt((a**2 - R**2) ** 2 + 2 * (a**2 + R**2) * z**2 + z**4)
                )
            )
            * (
                -3 * a**2
                + R**2
                + z**2
                + 3 * np.sqrt((a**2 - R**2) ** 2 + 2 * (a**2 + R**2) * z**2 + z**4)
            )
        ) / (np.sqrt(2) * np.pi * (R**2 + z**2) ** 2) - (
            2 * M * (-(a**2 * (R**2 - 2 * z**2)) + 2 * (R**2 + z**2) ** 2) * np.pi / 2
        ) / (np.pi * (R**2 + z**2) ** 2.5)
    else:
        return (
            M
            * np.sqrt(
                np.abs(
                    a**2
                    - R**2
                    - z**2
                    + np.sqrt((a**2 - R**2) ** 2 + 2 * (a**2 + R**2) * z**2 + z**4)
                )
            )
            * (
                -3 * a**2
                + R**2
                + z**2
                + 3 * np.sqrt((a**2 - R**2) ** 2 + 2 * (a**2 + R**2) * z**2 + z**4)
            )
        ) / (np.sqrt(2) * np.pi * (R**2 + z**2) ** 2) - (
            2
            * M
            * (-(a**2 * (R**2 - 2 * z**2)) + 2 * (R**2 + z**2) ** 2)
            * np.arctan(
                np.sqrt(2)
                * np.sqrt(
                    (R**2 + z**2)
                    / np.abs(
                        a**2
                        - R**2
                        - z**2
                        + np.sqrt((a**2 - R**2) ** 2 + 2 * (a**2 + R**2) * z**2 + z**4)
                    )
                )
            )
        ) / (np.pi * (R**2 + z**2) ** 2.5)


def nu_i(rho, z, M, MD, b, m):
    """Observer-frame nu potential (plain Python mirror of `nu`, used only for fsolve)."""
    nuSch = math.log(
        (d1_i(rho, z, M) + d2_i(rho, z, M) - 2 * M) / (d1_i(rho, z, M) + d2_i(rho, z, M) + 2 * M)
    )
    if m == 0:
        return nuSch
    elif m == 1:
        return nuD(rho, z, MD, b)
    elif m == 2:
        return nuSch + nuD_i(rho, z, MD, b)


def gpp_i(rho, z, M, MD, b):
    """Observer-frame g_phiphi metric component, used to solve for the initial observer rho."""
    return rho**2 * math.exp(-nu_i(rho, z, M, MD, b, 2))


def solve_observer_rho(M, MD, b, z0=0.0, target_gpp_sqrt=15.0, R_initial_guess=10.0):
    """Solve for the initial observer's rho such that sqrt(gpp) equals `target_gpp_sqrt`."""
    func_initial = lambda R: np.sqrt(gpp_i(float(R[0]), z0, M, MD, b)) - target_gpp_sqrt  # noqa: E731
    R_solution = fsolve(func_initial, R_initial_guess)
    return float(R_solution[0])


# ---------------------------------------------------------------------------
# Lambda potential: derivatives, quadrature tabulation, and lookup-matrix interpolation
# ---------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def derNU(rho, z, M, MD, b, l, m, hder):
    return derivative(nu, l, rho, z, hder, M, MD, b, m)


@jit(nopython=True, parallel=True)
def dlamb(z, rho, M, MD, b, l, m, hder):
    if l == 0:
        return (
            0.5
            * rho
            * (derNU(rho, z, M, MD, b, 0, m, hder) ** 2 - derNU(rho, z, M, MD, b, 1, m, hder) ** 2)
        )
    elif l == 1:
        return rho * derNU(rho, z, M, MD, b, 0, m, hder) * derNU(rho, z, M, MD, b, 1, m, hder)


@jit(nopython=True, parallel=True)
def dlamb2(rho, z, M, MD, b, l, m, hder):
    if l == 0:
        return (
            0.5
            * rho
            * (derNU(rho, z, M, MD, b, 0, m, hder) ** 2 - derNU(rho, z, M, MD, b, 1, m, hder) ** 2)
        )
    elif l == 1:
        return rho * derNU(rho, z, M, MD, b, 0, m, hder) * derNU(rho, z, M, MD, b, 1, m, hder)


@jit(forceobj=True)
def lamb_Mat(rho, z, M, MD, b, m, hder):
    """Tabulate the non-linear lambda potential by quadrature of dlamb/dlamb2.

    Object-mode jit (scipy.integrate.quad is not numba-compilable in nopython mode).
    Only used by generate_matriz.py to build the lookup matrix consumed by `lamb`.
    """
    limit = 0.0
    if z >= limit:
        return (
            sci.quad(dlamb, 40.0, z, (rho, M, MD, b, 1, m, hder))[0]
            + sci.quad(dlamb2, 40.0, rho, (40.0, M, MD, b, 0, m, hder))[0]
            + lambSch(40.0, 40.0, M + MD, MD, b)
        )
    elif z < -limit:
        return (
            sci.quad(dlamb, -40, z, (rho, M, MD, b, 1, m, hder))[0]
            + sci.quad(dlamb2, 40.0, rho, (-40.0, M, MD, b, 0, m, hder))[0]
            + lambSch(40.0, -40.0, M + MD, MD, b)
        )


@jit(nopython=True, parallel=True)
def lamb(rho, z, M, MD, b, m):
    """Bilinear interpolation of the lambda potential from the loaded `Mat_nu` lookup table.

    `m` is accepted for signature-compatibility with `nu` but not otherwise used: the
    lookup table already encodes the combined BH+disk (m=2) potential it was tabulated with.
    """
    zref = 40
    Rref = 40

    iR = (zref - z) / zref * (len(Mat_nu) - 1) / 2
    jR = (Rref - rho) / (Rref - 0.0) * (len(Mat_nu[0]) - 1)

    i0 = int(math.floor((zref - z) / zref * (len(Mat_nu) - 1) / 2))
    j0 = int(math.floor((Rref - rho) / (Rref - 0.0) * (len(Mat_nu[0]) - 1)))

    if np.abs(jR) > len(Mat_nu[0]) - 1:
        iR = int(len(Mat_nu) / 2) + 0.5
        jR = len(Mat_nu[0]) - 0.5
        i0 = int(math.floor(iR))
        j0 = int(math.floor(jR))

    I = iR - i0
    J = jR - j0

    f00 = Mat_nu[i0, j0]
    f01 = Mat_nu[i0, j0 + 1]
    f10 = Mat_nu[i0 + 1, j0]
    f11 = Mat_nu[i0 + 1, j0 + 1]

    return (f00 + (f01 - f00) * J) + (f10 - f00 + (f11 + f00 - f01 - f10) * J) * I


# ---------------------------------------------------------------------------
# Metric components and derived geodesic quantities
# ---------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def gtt(rho, z, M, MD, b):
    return -math.exp(nu(rho, z, M, MD, b, 2))


@jit(nopython=True, parallel=True)
def grr(rho, z, M, MD, b):
    return math.exp(lamb(rho, z, M, MD, b, 2) - nu(rho, z, M, MD, b, 2))


@jit(nopython=True, parallel=True)
def gzz(rho, z, M, MD, b):
    return math.exp(lamb(rho, z, M, MD, b, 2) - nu(rho, z, M, MD, b, 2))


@jit(nopython=True, parallel=True)
def gpp(rho, z, M, MD, b):
    return rho**2 * math.exp(-nu(rho, z, M, MD, b, 2))


@jit(nopython=True, parallel=True)
def zeta(rho, z, M, MD, b):
    return np.sqrt(-1 / gtt(rho, z, M, MD, b))


@jit(nopython=True, parallel=True)
def dthe(rho, z, M, MD, b, alfa):
    return 1 / np.sqrt(gzz(rho, z, M, MD, b)) * np.sin(alfa)


@jit(nopython=True, parallel=True)
def dr(rho, z, M, MD, b, alfa, beta):
    return 1 / np.sqrt(grr(rho, z, M, MD, b)) * np.cos(alfa) * np.cos(beta)


@jit(nopython=True, parallel=True)
def Pphi(rho, z, M, MD, b, alfa, beta):
    return np.sqrt(gpp(rho, z, M, MD, b)) * np.sin(beta) * np.cos(alfa)


@jit(nopython=True, parallel=True)
def Pt(rho, z, M, MD, b, alfa, beta):
    return -1 / zeta(rho, z, M, MD, b)


@jit(nopython=True, parallel=True)
def dphi(rho, z, M, MD, b, alfa, beta, p_phi):
    return 1 / gpp(rho, z, M, MD, b) * p_phi


@jit(nopython=True, parallel=True)
def dt(rho, z, M, MD, b, alfa, beta, p_t):
    return 1 / gtt(rho, z, M, MD, b) * p_t


# ---------------------------------------------------------------------------
# RKF45 (Runge-Kutta-Fehlberg) adaptive-step integrator.
#
# Coefficients and step-control logic are untouched from the legacy implementation.
# ---------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def run_kut4_mod(F, x, y, h, *args):
    hmax = -0.04
    hmin = -(10**-7)
    tol = 10**-4

    K1 = h * F(x, y, *args)
    K2 = h * F(x + h / 4, y + K1 / 4, *args)
    K3 = h * F(x + 3 / 8 * h, y + 3 / 32 * K1 + 9 / 32 * K2, *args)
    K4 = h * F(x + 12 / 13 * h, y + 1932 / 2197 * K1 - 7200 / 2197 * K2 + 7296 / 2197 * K3, *args)
    K5 = h * F(x + h, y + 439 / 216 * K1 - 8 * K2 + 3680 / 513 * K3 - 845 / 4104 * K4, *args)
    K6 = h * F(
        x + h / 2,
        y - 8 / 27 * K1 + 2 * K2 - 3544 / 2565 * K3 + 1859 / 4104 * K4 - 11 / 40 * K5,
        *args,
    )

    y4 = y + 25 / 216 * K1 + 1408 / 2565 * K3 + 2197 / 4101 * K4 - K5 / 5
    y5 = y + 16 / 135 * K1 + 6656 / 12825 * K3 + 28561 / 56430 * K4 - 9 / 50 * K5 + 2 / 55 * K6

    error = np.linalg.norm(y5 - y4)
    delta = pow(1.0 / 2.0, (1.0 / 4.0)) * pow(tol / error, (1.0 / 4.0))

    if error < tol:
        if np.abs(delta * h) < np.abs(hmin):
            h = hmin
        elif np.abs(delta * h) > np.abs(hmax):
            h = hmax
        else:
            h = delta * h

        x = x + h
        y = y4

    else:
        it = 0
        while error > tol:
            h = delta * h
            it += 1

            K1 = h * F(x, y, *args)
            K2 = h * F(x + h / 4, y + K1 / 4, *args)
            K3 = h * F(x + 3 / 8 * h, y + 3 / 32 * K1 + 9 / 32 * K2, *args)
            K4 = h * F(
                x + 12 / 13 * h, y + 1932 / 2197 * K1 - 7200 / 2197 * K2 + 7296 / 2197 * K3, *args
            )
            K5 = h * F(
                x + h, y + 439 / 216 * K1 - 8 * K2 + 3680 / 513 * K3 - 845 / 4104 * K4, *args
            )
            K6 = h * F(
                x + h / 2,
                y - 8 / 27 * K1 + 2 * K2 - 3544 / 2565 * K3 + 1859 / 4104 * K4 - 11 / 40 * K5,
                *args,
            )

            y4 = y + 25 / 216 * K1 + 1408 / 2565 * K3 + 2197 / 4101 * K4 - K5 / 5
            y5 = (
                y
                + 16 / 135 * K1
                + 6656 / 12825 * K3
                + 28561 / 56430 * K4
                - 9 / 50 * K5
                + 2 / 55 * K6
            )

            error = np.linalg.norm(y5 - y4)
            delta = pow(1.0 / 2.0, (1.0 / 4.0)) * pow(tol / error, (1.0 / 4.0))

            if error < tol:
                if h < hmin:
                    h = hmin
                    x = x + h
                    y = y4
                else:
                    x = x + h
                    y = y4
            elif it > 500:
                if h < hmin:
                    h = hmin
                    x = x + h
                    y = y4

    return (h, x, y)


# ---------------------------------------------------------------------------
# Geodesic RHS + single-ray tracers.
#
# Two state-vector variants exist:
#  - geo/func: 4-state [rho, drho, z, dz] (no phi tracking) -- used by the parallel and
#    serial shadow tracers (test_parallel_SHADOW.py, test_Z_SHADOW.py).
#  - geo_lensing/func_lensing: 5-state [rho, drho, z, dz, phi] -- used by the lensing
#    driver (test_symmetry_lensing.py) which also needs the accumulated phi coordinate.
# ---------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def geo(t, z, M, alfa, beta, rho0, z0, MD, b, hder):
    """Geodesic RHS for the 4-state [rho, drho, z, dz] system (no phi tracking)."""
    pphi = Pphi(rho0, z0, M, MD, b, alfa, beta)
    pt = Pt(rho0, z0, M, MD, b, alfa, beta)

    d2Rdt = -(
        0.5
        * math.exp(2 * nu(z[0], z[2], M, MD, b, 2) - lamb(z[0], z[2], M, MD, b, 2))
        * derNU(z[0], z[2], M, MD, b, 0, 2, hder)
        * dt(z[0], z[2], M, MD, b, alfa, beta, pt) ** 2
        + 0.5
        * (dlamb2(z[0], z[2], M, MD, b, 0, 2, hder) - derNU(z[0], z[2], M, MD, b, 0, 2, hder))
        * z[1] ** 2
        + (dlamb2(z[0], z[2], M, MD, b, 1, 2, hder) - derNU(z[0], z[2], M, MD, b, 1, 2, hder))
        * z[1]
        * z[3]
        - 0.5
        * (dlamb2(z[0], z[2], M, MD, b, 0, 2, hder) - derNU(z[0], z[2], M, MD, b, 0, 2, hder))
        * z[3] ** 2
        + 0.5
        * math.exp(-lamb(z[0], z[2], M, MD, b, 2))
        * z[0]
        * (-2 + z[0] * derNU(z[0], z[2], M, MD, b, 0, 2, hder))
        * dphi(z[0], z[2], M, MD, b, alfa, beta, pphi) ** 2
    )

    d2Thedt = -(
        0.5
        * math.exp(2 * nu(z[0], z[2], M, MD, b, 2) - lamb(z[0], z[2], M, MD, b, 2))
        * derNU(z[0], z[2], M, MD, b, 1, 2, hder)
        * dt(z[0], z[2], M, MD, b, alfa, beta, pt) ** 2
        - 0.5
        * (dlamb2(z[0], z[2], M, MD, b, 1, 2, hder) - derNU(z[0], z[2], M, MD, b, 1, 2, hder))
        * z[1] ** 2
        + (dlamb2(z[0], z[2], M, MD, b, 0, 2, hder) - derNU(z[0], z[2], M, MD, b, 0, 2, hder))
        * z[1]
        * z[3]
        + 0.5
        * (dlamb2(z[0], z[2], M, MD, b, 1, 2, hder) - derNU(z[0], z[2], M, MD, b, 1, 2, hder))
        * z[3] ** 2
        + 0.5
        * math.exp(-lamb(z[0], z[2], M, MD, b, 2))
        * z[0] ** 2
        * derNU(z[0], z[2], M, MD, b, 1, 2, hder)
        * dphi(z[0], z[2], M, MD, b, alfa, beta, pphi) ** 2
    )

    return np.array([z[1], d2Rdt, z[3], d2Thedt])


@jit(nopython=True, parallel=True)
def func(y, x, h, alfa, beta, M, rho0, z0, MD, b, hder, disk_visible):
    """Trace a single null geodesic (4-state) until it hits the BH, escapes, or crosses the disk.

    Returns the final [rho, z] pair. The [0.001, 0.001] sentinel marks BH capture. With
    `disk_visible=True`, a ray that crosses the disk plane outside rho = b stops with its z
    offset by +50.0 (the disk-hit sentinel). With `disk_visible=False` the disk is ignored.
    """
    y_prev = y
    yf = [y[0], y[2]]

    while (
        nu(y[0], y[2], M, MD, b, 2) > -3.0
        and np.sqrt(gpp(np.sqrt(y[0] ** 2 + y[2] ** 2), 0, M, MD, b)) < 30.0
    ):
        (h, x, y) = run_kut4_mod(geo, x, y, h, M, alfa, beta, rho0, z0, MD, b, hder)

        if np.sqrt(gpp(np.sqrt(y[0] ** 2 + y[2] ** 2), 0, M, MD, b)) >= 30.0:
            yf = [y[0], y[2]]
            break

        elif nu(y[0], y[2], M, MD, b, 2) <= -3.0:
            yf = [0.001, 0.001]
            break

        elif disk_visible and y[0] > b and np.sign(y[2]) == -np.sign(y_prev[2]):
            yf = [y[0], y[2] + 50.0]
            break

        y_prev = y

    return yf


@jit(nopython=True, parallel=True)
def func_bh_disk_crossing(y, x, h, alfa, beta, M, rho0, z0, MD, b, hder):
    """`func` with the disk always visible (kept for test_Z_SHADOW.py)."""
    return func(y, x, h, alfa, beta, M, rho0, z0, MD, b, hder, True)


@jit(nopython=True, parallel=True)
def geo_lensing(t, z, M, alfa, beta, rho0, z0, MD, b, hder):
    """Geodesic RHS for the 5-state [rho, drho, z, dz, phi] system (lensing variant).

    Same rho/z equations of motion as `geo`; adds dphi/dlambda so the accumulated phi
    coordinate can be reported (needed to classify lensed image quadrants).
    """
    pphi = Pphi(rho0, z0, M, MD, b, alfa, beta)
    pt = Pt(rho0, z0, M, MD, b, alfa, beta)

    d2Rdt = -(
        0.5
        * math.exp(2 * nu(z[0], z[2], M, MD, b, 2) - lamb(z[0], z[2], M, MD, b, 2))
        * derNU(z[0], z[2], M, MD, b, 0, 2, hder)
        * dt(z[0], z[2], M, MD, b, alfa, beta, pt) ** 2
        + 0.5
        * (dlamb2(z[0], z[2], M, MD, b, 0, 2, hder) - derNU(z[0], z[2], M, MD, b, 0, 2, hder))
        * z[1] ** 2
        + (dlamb2(z[0], z[2], M, MD, b, 1, 2, hder) - derNU(z[0], z[2], M, MD, b, 1, 2, hder))
        * z[1]
        * z[3]
        - 0.5
        * (dlamb2(z[0], z[2], M, MD, b, 0, 2, hder) - derNU(z[0], z[2], M, MD, b, 0, 2, hder))
        * z[3] ** 2
        + 0.5
        * math.exp(-lamb(z[0], z[2], M, MD, b, 2))
        * z[0]
        * (-2 + z[0] * derNU(z[0], z[2], M, MD, b, 0, 2, hder))
        * dphi(z[0], z[2], M, MD, b, alfa, beta, pphi) ** 2
    )

    d2Thedt = -(
        0.5
        * math.exp(2 * nu(z[0], z[2], M, MD, b, 2) - lamb(z[0], z[2], M, MD, b, 2))
        * derNU(z[0], z[2], M, MD, b, 1, 2, hder)
        * dt(z[0], z[2], M, MD, b, alfa, beta, pt) ** 2
        - 0.5
        * (dlamb2(z[0], z[2], M, MD, b, 1, 2, hder) - derNU(z[0], z[2], M, MD, b, 1, 2, hder))
        * z[1] ** 2
        + (dlamb2(z[0], z[2], M, MD, b, 0, 2, hder) - derNU(z[0], z[2], M, MD, b, 0, 2, hder))
        * z[1]
        * z[3]
        + 0.5
        * (dlamb2(z[0], z[2], M, MD, b, 1, 2, hder) - derNU(z[0], z[2], M, MD, b, 1, 2, hder))
        * z[3] ** 2
        + 0.5
        * math.exp(-lamb(z[0], z[2], M, MD, b, 2))
        * z[0] ** 2
        * derNU(z[0], z[2], M, MD, b, 1, 2, hder)
        * dphi(z[0], z[2], M, MD, b, alfa, beta, pphi) ** 2
    )

    dPhidt = 1 / gpp(z[0], z[2], M, MD, b) * pphi

    return np.array([z[1], d2Rdt, z[3], d2Thedt, dPhidt])


@jit(nopython=True, parallel=True)
def func_lensing(y, x, h, alfa, beta, M, rho0, z0, MD, b, hder):
    """Trace a single null geodesic (5-state, with phi) for the lensing driver.

    Returns the final [rho, z, phi] triple.
    """
    Y = [y]

    while (
        nu(y[0], y[2], M, MD, b, 2) > -3.0
        and np.sqrt(gpp(np.sqrt(y[0] ** 2 + y[2] ** 2), 0, M, MD, b)) < 30.0
    ):
        (h, x, y) = run_kut4_mod(geo_lensing, x, y, h, M, alfa, beta, rho0, z0, MD, b, hder)
        Y.append(y)

        if np.sqrt(gpp(np.sqrt(y[0] ** 2 + y[2] ** 2), 0, M, MD, b)) >= 30.0:
            yf = [Y[-1][0], Y[-1][2], Y[-1][4]]
            break

        elif nu(y[0], y[2], M, MD, b, 2) <= -3.0:
            yf = [0.001, 0.001, 0.2]
            break

    return yf


@jit(nopython=True, parallel=True)
def f_paral(rho0, z0, M, MD, b, alfa, beta, hder, disk_visible=False):
    """Production multicore driver: trace one ray per (alfa, beta) grid point in parallel.

    Uses numba's `prange` (requires `parallel=True`, see module docstring) to
    auto-parallelize the two nested angle loops.
    """
    Mat = np.zeros((len(alfa), len(beta)))
    Mz = np.zeros((len(alfa), len(beta)))

    for i in prange(len(alfa)):
        for j in prange(len(beta)):
            y = np.array(
                [
                    rho0,
                    dr(rho0, z0, M, MD, b, alfa[i], beta[j]),
                    z0,
                    dthe(rho0, z0, M, MD, b, alfa[i]),
                ]
            )
            (Mat[i, j], Mz[i, j]) = func(
                y, 300.0, -0.02, alfa[i], beta[j], M, rho0, z0, MD, b, hder, disk_visible
            )

    return (Mat, Mz)
