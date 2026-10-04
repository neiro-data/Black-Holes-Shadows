"""Physics core of the Weyl family: Schwarzschild black hole with an optional Morgan-Morgan disk.

The runner and the GPU port use this module.

Shared parameters of the helpers:

- `rho`, `z`: Weyl coordinates.
- `M`, `MD`, `b`: black-hole mass, disk mass and disk inner radius.
- `alpha`, `beta`: pixel angles of the ray.
- `l`: derivative axis. 0 is rho, 1 is z.
- `m`: potential selector. 0 is the black hole only (`nuSch`), 1 is the disk only (`nuD`), 2 is
  both. The geodesic code always uses 2.
- `hder`: finite-difference step.

Rules. Do not change them without a new derivation of the physics:

- **`xi2` `np.abs` guard.** Rounding can make the radicand slightly negative near the disk plane.
  Without the guard, numba raises an error.
- **`Mat_nu` global.** `lamb` reads the lambda table from the global `Mat_nu`. Numba fixes the
  global at the first compile. Call `load_matrix(path)` before the first trace. The table is not
  a function argument, because each RK step would then pass a large array.
- **`parallel=True` only on `f_paral` and `retrace_points`.** Only these functions have a
  `prange` loop. On the scalar helpers, the flag only gives `NumbaPerformanceWarning`.
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
    """Load the lambda table into the global `Mat_nu`.

    Call this function before the first trace. Numba fixes `Mat_nu` when it first compiles a
    function that reads it.

    Parameters
    ----------
    path : str
        `.npz` file with the key `lam` (from `bh_shadows.core.matrix`).
    """
    global Mat_nu
    with np.load(path) as f:
        Mat_nu = f["lam"]


# ---------------------------------------------------------------------------
# Numerical integration / differentiation helpers
# ---------------------------------------------------------------------------
@jit(nopython=True)
def derivative(f, l, x, y, hder, *args):
    """Calculate a central finite difference of `f`.

    Parameters
    ----------
    f : callable
        Function `f(x, y, *args)`.
    l : {0, 1}
        0 for d/dx, 1 for d/dy.
    x, y : float
        Evaluation point.
    hder : float
        Step. The function uses `abs(hder)`.
    *args
        Extra arguments of `f`.

    Returns
    -------
    float
        Derivative value.
    """
    if l == 0:
        return 0.5 * (f(x + np.abs(hder), y, *args) - f(x - np.abs(hder), y, *args)) / np.abs(hder)
    elif l == 1:
        return 0.5 * (f(x, y + np.abs(hder), *args) - f(x, y - np.abs(hder), *args)) / np.abs(hder)
    elif l != 0 and l != 1:
        print("Error. Check 'l' number")


# ---------------------------------------------------------------------------
# Weyl-coordinate distances to the two Schwarzschild "rod" endpoints (+/-M on the axis)
# ---------------------------------------------------------------------------
@jit(nopython=True)
def d1(rho, z, M):
    """Distance from (rho, z) to the rod end at z = +M."""
    return np.sqrt(rho**2 + (z - M) ** 2)


@jit(nopython=True)
def d2(rho, z, M):
    """Distance from (rho, z) to the rod end at z = -M."""
    return np.sqrt(rho**2 + (z + M) ** 2)


# ---------------------------------------------------------------------------
# Inverted Morgan-Morgan disk potentials
# ---------------------------------------------------------------------------
@jit(nopython=True)
def xi2(R, z, a):
    """Return the spheroidal coordinate of the Morgan-Morgan disk potential.

    The `np.abs` guard is necessary. Rounding can make the radicand slightly negative near z = 0,
    R >= a. Without the guard, numba raises an error.

    Parameters
    ----------
    R, z : float
        Weyl coordinates.
    a : float
        Disk inner radius.

    Returns
    -------
    float
        Coordinate value.
    """
    return np.sqrt(
        np.abs(
            (a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2) ** 2 + 2 * (a**2 + R**2) * z**2 + z**4))
            / (R**2 + z**2)
        )
    ) / np.sqrt(2)


@jit(nopython=True)
def nuD(R, z, M, a):
    """Return the Morgan-Morgan (inverted) disk part of the nu potential.

    Parameters
    ----------
    R, z : float
        Weyl coordinates.
    M : float
        Disk mass.
    a : float
        Disk inner radius.

    Returns
    -------
    float
        Potential value.
    """
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
@jit(nopython=True)
def nu(rho, z, M, MD, b, m):
    """Return the nu potential.

    Parameters
    ----------
    rho, z : float
        Weyl coordinates.
    M, MD, b : float
        Black-hole mass, disk mass and disk inner radius.
    m : {0, 1, 2}
        0 black hole only, 1 disk only, 2 black hole and disk.

    Returns
    -------
    float
        Potential value.
    """
    nuSch = math.log(
        (d1(rho, z, M) + d2(rho, z, M) - 2 * M) / (d1(rho, z, M) + d2(rho, z, M) + 2 * M)
    )
    if m == 0:
        return nuSch
    elif m == 1:
        return nuD(rho, z, MD, b)
    elif m == 2:
        return nuSch + nuD(rho, z, MD, b)


@jit(nopython=True)
def lambSch(rho, z, M, MD, b):
    """Return the analytic Schwarzschild lambda. `lamb_Mat` uses it as the outer boundary value."""
    sigma = np.sqrt((rho**2 + z**2 + M**2) ** 2 - 4 * z**2 * M**2)
    return np.log(((d1(rho, z, M) + d2(rho, z, M)) ** 2 - 4 * M**2) / (4 * sigma))


# ---------------------------------------------------------------------------
# Observer-frame (`_i`) variants: plain Python (not numba), used only once to solve
# for the initial observer's rho via `scipy.optimize.fsolve`.
# ---------------------------------------------------------------------------
def d1_i(rho, z, M):
    """Python version of `d1` (no numba). For `fsolve` only."""
    return np.sqrt(rho**2 + (z - M) ** 2)


def d2_i(rho, z, M):
    """Python version of `d2` (no numba). For `fsolve` only."""
    return np.sqrt(rho**2 + (z + M) ** 2)


def xi2_i(R, z, a):
    """Python version of `xi2` without the `np.abs` guard. For `fsolve` only."""
    return np.sqrt(
        (a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2) ** 2 + 2 * (a**2 + R**2) * z**2 + z**4))
        / (R**2 + z**2)
    ) / np.sqrt(2)


def nuD_i(R, z, M, a):
    """Python version of `nuD` (no numba). For `fsolve` only."""
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
    """Python version of `nu` (no numba). For `fsolve` only."""
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
    """Python version of `gpp` (no numba). For `fsolve` only."""
    return rho**2 * math.exp(-nu_i(rho, z, M, MD, b, 2))


def solve_observer_rho(M, MD, b, z0=0.0, target_gpp_sqrt=15.0, R_initial_guess=10.0):
    """Find the observer rho where `sqrt(gpp)` has a given value.

    Parameters
    ----------
    M, MD, b : float
        Black-hole mass, disk mass and disk inner radius.
    z0 : float, optional
        Observer z.
    target_gpp_sqrt : float, optional
        Value of `sqrt(gpp)` (the circumferential radius of the observer).
    R_initial_guess : float, optional
        Start value for `fsolve`.

    Returns
    -------
    float
        Observer rho.
    """
    func_initial = lambda R: np.sqrt(gpp_i(float(R[0]), z0, M, MD, b)) - target_gpp_sqrt  # noqa: E731
    R_solution = fsolve(func_initial, R_initial_guess)
    return float(R_solution[0])


# ---------------------------------------------------------------------------
# Lambda potential: derivatives, quadrature tabulation, and lookup-matrix interpolation
# ---------------------------------------------------------------------------
@jit(nopython=True)
def derNU(rho, z, M, MD, b, l, m, hder):
    """Return d nu / d rho (`l = 0`) or d nu / d z (`l = 1`) by central difference."""
    return derivative(nu, l, rho, z, hder, M, MD, b, m)


@jit(nopython=True)
def dlamb(z, rho, M, MD, b, l, m, hder):
    """Return component `l` of grad lambda (vacuum field equations). Order `(z, rho)` for `quad`."""
    if l == 0:
        return (
            0.5
            * rho
            * (derNU(rho, z, M, MD, b, 0, m, hder) ** 2 - derNU(rho, z, M, MD, b, 1, m, hder) ** 2)
        )
    elif l == 1:
        return rho * derNU(rho, z, M, MD, b, 0, m, hder) * derNU(rho, z, M, MD, b, 1, m, hder)


@jit(nopython=True)
def dlamb2(rho, z, M, MD, b, l, m, hder):
    """Return component `l` of grad lambda (vacuum field equations). Order `(rho, z)`."""
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
    """Calculate lambda at one point by quadrature of grad lambda.

    The function integrates from the outer boundary (|z| = 40, rho = 40), where lambda has the
    Schwarzschild value. It uses object mode, because numba cannot compile `scipy.integrate.quad`.
    Only `bh_shadows.core.matrix` calls it.

    Parameters
    ----------
    rho, z : float
        Weyl coordinates.
    M, MD, b : float
        Black-hole mass, disk mass and disk inner radius.
    m : {0, 1, 2}
        Potential selector (see `nu`).
    hder : float
        Finite-difference step.

    Returns
    -------
    float
        Lambda value.
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


@jit(nopython=True)
def lamb(rho, z, M, MD, b, m):
    """Return lambda by bilinear interpolation in `Mat_nu`.

    Parameters
    ----------
    rho, z : float
        Weyl coordinates.
    M, MD, b : float
        Not used. The table already holds the potential.
    m : int
        Not used. It keeps the same signature as `nu`.

    Returns
    -------
    float
        Lambda value.
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
@jit(nopython=True)
def gtt(rho, z, M, MD, b):
    """Return the metric component g_tt."""
    return -math.exp(nu(rho, z, M, MD, b, 2))


@jit(nopython=True)
def grr(rho, z, M, MD, b):
    """Return the metric component g_rhorho."""
    return math.exp(lamb(rho, z, M, MD, b, 2) - nu(rho, z, M, MD, b, 2))


@jit(nopython=True)
def gzz(rho, z, M, MD, b):
    """Return the metric component g_zz."""
    return math.exp(lamb(rho, z, M, MD, b, 2) - nu(rho, z, M, MD, b, 2))


@jit(nopython=True)
def gpp(rho, z, M, MD, b):
    """Return the metric component g_phiphi."""
    return rho**2 * math.exp(-nu(rho, z, M, MD, b, 2))


@jit(nopython=True)
def zeta(rho, z, M, MD, b):
    """Return `sqrt(-1 / g_tt)`, the lapse factor of the static observer."""
    return np.sqrt(-1 / gtt(rho, z, M, MD, b))


@jit(nopython=True)
def dthe(rho, z, M, MD, b, alpha):
    """Return the initial dz / dlambda of the ray for pixel angle `alpha`."""
    return 1 / np.sqrt(gzz(rho, z, M, MD, b)) * np.sin(alpha)


@jit(nopython=True)
def dr(rho, z, M, MD, b, alpha, beta):
    """Return the initial drho / dlambda of the ray for pixel angles `alpha`, `beta`."""
    return 1 / np.sqrt(grr(rho, z, M, MD, b)) * np.cos(alpha) * np.cos(beta)


@jit(nopython=True)
def Pphi(rho, z, M, MD, b, alpha, beta):
    """Return the conserved momentum p_phi of the ray."""
    return np.sqrt(gpp(rho, z, M, MD, b)) * np.sin(beta) * np.cos(alpha)


@jit(nopython=True)
def Pt(rho, z, M, MD, b, alpha, beta):
    """Return the conserved momentum p_t of the ray."""
    return -1 / zeta(rho, z, M, MD, b)


@jit(nopython=True)
def dphi(rho, z, M, MD, b, alpha, beta, p_phi):
    """Return dphi / dlambda from p_phi."""
    return 1 / gpp(rho, z, M, MD, b) * p_phi


@jit(nopython=True)
def dt(rho, z, M, MD, b, alpha, beta, p_t):
    """Return dt / dlambda from p_t."""
    return 1 / gtt(rho, z, M, MD, b) * p_t


# ---------------------------------------------------------------------------
# RKF45 (Runge-Kutta-Fehlberg) adaptive-step integrator.
#
# Coefficients and step-control logic are untouched from the legacy implementation.
# ---------------------------------------------------------------------------
@jit(nopython=True)
def run_kut4_mod(F, x, y, h, *args):
    """Do one adaptive RKF45 step.

    The step control is the legacy one: `hmax = -0.04`, `hmin = -1e-7`, `tol = 1e-4`. The retry loop
    compares `h < hmin` on negative steps and can do more than 500 retries. Keep it as it is: the
    CPU results depend on it.

    Parameters
    ----------
    F : callable
        Right-hand side `F(x, y, *args)`.
    x : float
        Affine parameter.
    y : np.ndarray
        State.
    h : float
        Step size (negative).
    *args
        Extra arguments of `F`.

    Returns
    -------
    h : float
        Next step size.
    x : float
        New affine parameter.
    y : np.ndarray
        New state.
    """
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
# State vector: [rho, drho, z, dz] (phi is not traced).
# ---------------------------------------------------------------------------
@jit(nopython=True)
def geo(t, z, M, alpha, beta, rho0, z0, MD, b, hder):
    """Return the geodesic right-hand side for the state `[rho, drho, z, dz]`.

    Parameters
    ----------
    t : float
        Affine parameter (not used).
    z : np.ndarray
        State `[rho, drho, z, dz]`.
    M : float
        Black-hole mass.
    alpha, beta : float
        Pixel angles.
    rho0, z0 : float
        Observer position.
    MD, b : float
        Disk mass and disk inner radius.
    hder : float
        Finite-difference step.

    Returns
    -------
    np.ndarray
        Derivative of the state.
    """
    pphi = Pphi(rho0, z0, M, MD, b, alpha, beta)
    pt = Pt(rho0, z0, M, MD, b, alpha, beta)

    d2Rdt = -(
        0.5
        * math.exp(2 * nu(z[0], z[2], M, MD, b, 2) - lamb(z[0], z[2], M, MD, b, 2))
        * derNU(z[0], z[2], M, MD, b, 0, 2, hder)
        * dt(z[0], z[2], M, MD, b, alpha, beta, pt) ** 2
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
        * dphi(z[0], z[2], M, MD, b, alpha, beta, pphi) ** 2
    )

    d2Thedt = -(
        0.5
        * math.exp(2 * nu(z[0], z[2], M, MD, b, 2) - lamb(z[0], z[2], M, MD, b, 2))
        * derNU(z[0], z[2], M, MD, b, 1, 2, hder)
        * dt(z[0], z[2], M, MD, b, alpha, beta, pt) ** 2
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
        * dphi(z[0], z[2], M, MD, b, alpha, beta, pphi) ** 2
    )

    return np.array([z[1], d2Rdt, z[3], d2Thedt])


@jit(nopython=True)
def func(y, x, h, alpha, beta, M, rho0, z0, MD, b, hder, disk_visible):
    """Trace one ray until it is captured, escapes or hits the disk.

    Parameters
    ----------
    y : np.ndarray
        Initial state `[rho, drho, z, dz]`.
    x : float
        Initial affine parameter.
    h : float
        Initial step (negative: the ray goes back in time).
    alpha, beta : float
        Pixel angles.
    M : float
        Black-hole mass.
    rho0, z0 : float
        Observer position.
    MD, b : float
        Disk mass and disk inner radius.
    hder : float
        Finite-difference step.
    disk_visible : bool
        Stop the ray when it crosses the disk plane at rho > b.

    Returns
    -------
    list of float
        End `[rho, z]`. Sentinels: capture gives `[0.001, 0.001]`, a disk hit gives `[rho, z + 50]`.
    """
    y_prev = y
    yf = [y[0], y[2]]

    while (
        nu(y[0], y[2], M, MD, b, 2) > -3.0
        and np.sqrt(gpp(np.sqrt(y[0] ** 2 + y[2] ** 2), 0, M, MD, b)) < 30.0
    ):
        (h, x, y) = run_kut4_mod(geo, x, y, h, M, alpha, beta, rho0, z0, MD, b, hder)

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
def f_paral(rho0, z0, M, MD, b, alpha, beta, hder, disk_visible=False):
    """Trace one ray for each pixel on all CPU cores (numba `prange`).

    Parameters
    ----------
    rho0, z0 : float
        Observer position.
    M, MD, b : float
        Black-hole mass, disk mass and disk inner radius.
    alpha, beta : np.ndarray
        Pixel angles.
    hder : float
        Finite-difference step.
    disk_visible : bool, optional
        Stop rays at the disk.

    Returns
    -------
    Mat, Mz : np.ndarray
        End rho and end z, of shape (len(alpha), len(beta)). See `func` for the sentinels.
    """
    Mat = np.zeros((len(alpha), len(beta)))
    Mz = np.zeros((len(alpha), len(beta)))

    for i in prange(len(alpha)):
        for j in prange(len(beta)):
            y = np.array(
                [
                    rho0,
                    dr(rho0, z0, M, MD, b, alpha[i], beta[j]),
                    z0,
                    dthe(rho0, z0, M, MD, b, alpha[i]),
                ]
            )
            (Mat[i, j], Mz[i, j]) = func(
                y, 300.0, -0.02, alpha[i], beta[j], M, rho0, z0, MD, b, hder, disk_visible
            )

    return (Mat, Mz)


@jit(nopython=True, parallel=True)
def retrace_points(rho0, z0, M, MD, b, alpha, beta, ii, jj, hder, disk_visible=False):
    """Trace selected pixels again in parallel. The rays are the same as in `f_paral`.

    Parameters
    ----------
    rho0, z0 : float
        Observer position.
    M, MD, b : float
        Black-hole mass, disk mass and disk inner radius.
    alpha, beta : np.ndarray
        Pixel angles of the full grid.
    ii, jj : np.ndarray
        Row and column indices of the k pixels.
    hder : float
        Finite-difference step.
    disk_visible : bool, optional
        Stop rays at the disk.

    Returns
    -------
    Mat, Mz : np.ndarray
        End rho and end z, of shape (k,).
    """
    k = len(ii)
    Mat = np.zeros(k)
    Mz = np.zeros(k)

    for p in prange(k):
        i = ii[p]
        j = jj[p]
        y = np.array(
            [
                rho0,
                dr(rho0, z0, M, MD, b, alpha[i], beta[j]),
                z0,
                dthe(rho0, z0, M, MD, b, alpha[i]),
            ]
        )
        (Mat[p], Mz[p]) = func(
            y, 300.0, -0.02, alpha[i], beta[j], M, rho0, z0, MD, b, hder, disk_visible
        )

    return (Mat, Mz)
