#!/usr/bin/env python
# coding: utf-8
"""Shared Weyl-coordinate physics core for the Schwarzschild BH + Morgan-Morgan
disk ray tracers in this repository (generate_matriz.py, test_parallel_SHADOW.py,
test_Z_SHADOW.py, test_symmetry_lensing.py).

This module holds the helpers that were previously copy-pasted near-verbatim
across those four scripts (tagged `#Duplicated` in their history) -- numerical
utilities, the nu/lambda metric potentials, the metric components and photon
momenta/velocities, the RKF45 integrator, and the non-jitted observer-frame
variants used to locate the initial observer position via fsolve.

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
pre-tabulated lambda matrix held in the module-level `Mat_nu` global. Call
`load_matrix(path)` once at script startup (before any `lamb` call) to set it.
This mirrors the previous per-script `Mat_nu = np.loadtxt(...)` pattern; a
more explicit (parameter-threaded) design is a possible follow-up.
"""

import math
import numpy as np
from numba import jit

__all__ = [
    "simps", "derivative", "run_kut4_mod",
    "d1", "d2", "xi2", "nuD", "nu", "lambSch",
    "gtt", "grr", "gzz", "gpp", "zeta",
    "dthe", "dr", "Pphi", "Pt", "dphi", "dt",
    "derNU", "dlamb", "dlamb2", "lamb", "load_matrix",
    "d1_i", "d2_i", "xi2_i", "nuD_i", "nu_i", "gpp_i",
]

# Bilinear-interpolation source matrix for `lamb`; set via `load_matrix`
# before any `lamb`/grr/gzz/dr/dthe/geo call. See module docstring.
Mat_nu = None


def load_matrix(path):
    """Load the pre-tabulated lambda-potential matrix (produced by
    generate_matriz.py) into the module-level `Mat_nu` used by `lamb`."""
    global Mat_nu
    Mat_nu = np.loadtxt(path)
    return Mat_nu


##########################
# Numerical utilities: integration and derivatives
@jit(nopython=True)
def simps(f, l, z, a, b, N, *args):
    """Composite Simpson's rule for f(x, z, *args) or f(z, x, *args).

    Integrates a 1-D slice of `f` over [a, b] using N (even) subintervals.

    Args:
        f: integrand, called as f(x, z, *args) if l == 0, or f(z, x, *args) if l == 1.
        l: 0 to integrate over the function's first argument, 1 for its second.
        z: the argument held fixed while integrating over the other one.
        a, b: integration limits for the swept argument.
        N: number of subintervals (must be even).
        *args: extra positional arguments forwarded to f.

    Returns:
        The Simpson's-rule estimate of the integral.
    """
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


@jit(nopython=True)
def derivative(f, l, x, y, hder, *args):
    """Central finite-difference derivative of f(x, y, *args).

    Args:
        f: function of two coordinates plus *args.
        l: 0 to differentiate with respect to x, 1 to differentiate with respect to y.
        x, y: point at which to evaluate the derivative.
        hder: finite-difference step size (its absolute value is used).
        *args: extra positional arguments forwarded to f.

    Returns:
        The central-difference approximation of df/dx or df/dy. Prints an
        error and implicitly returns None if l is neither 0 nor 1.
    """
    if l == 0:
        return (0.5 * (f(x + np.abs(hder), y, *args) - f(x - np.abs(hder), y, *args)) / np.abs(hder))

    elif l == 1:
        return (0.5 * (f(x, y + np.abs(hder), *args) - f(x, y - np.abs(hder), *args)) / np.abs(hder))

    elif (l != 0 and l != 1):
        print("Error. Check 'l' number")

##################################


@jit(nopython=True)
def d1(rho, z, M):
    """Distance from (rho, z) to the upper Weyl-coordinate rod endpoint (0, +M) of the Schwarzschild BH."""
    return np.sqrt(rho**2 + (z - M)**2)


@jit(nopython=True)
def d2(rho, z, M):
    """Distance from (rho, z) to the lower Weyl-coordinate rod endpoint (0, -M) of the Schwarzschild BH."""
    return np.sqrt(rho**2 + (z + M)**2)


@jit(nopython=True)
def xi2(R, z, a):
    """Oblate-spheroidal-like coordinate used in the Morgan-Morgan disk potential nuD, for disk parameter a.

    Canonical (np.abs-guarded) form: see module docstring.
    """
    return (np.sqrt(np.abs((a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)) / (R**2 + z**2))) / np.sqrt(2))


@jit(nopython=True)
def nuD(R, z, M, a):
    """Morgan-Morgan finite thin-disk contribution to the nu metric potential.

    Args:
        R, z: Weyl cylindrical coordinates.
        M: disk mass parameter.
        a: disk outer-edge (radius) parameter.

    Returns:
        nuD(R, z), handling the on-disk-plane singular case (xi2 == 0) separately.
    """
    # if (np.abs(z) < 10**-6 and R >= a):
    if xi2(R, z, a) == 0.0:
        # print("a")
        return ((M*np.sqrt(np.abs(a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))*(-3*a**2 + R**2 + z**2 + 3*np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))/(np.sqrt(2)*np.pi*(R**2 + z**2)**2) - (2*M*(-(a**2*(R**2 - 2*z**2)) + 2*(R**2 + z**2)**2)*np.pi/2)/(np.pi*(R**2 + z**2)**2.5))
    # if (xi2(R,z,a) == 0.0):
    #     return  (- (2*M*(-(a**2*(R**2 - 2*z**2)) + 2*(R**2 + z**2)**2)*np.pi/2)/(np.pi*(R**2 + z**2)**2.5) )
    else:
        # print("b")
        return ((M*np.sqrt(np.abs(a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))*(-3*a**2 + R**2 + z**2 + 3*np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))/(np.sqrt(2)*np.pi*(R**2 + z**2)**2) - (2*M*(-(a**2*(R**2 - 2*z**2)) + 2*(R**2 + z**2)**2)*np.arctan(np.sqrt(2)*np.sqrt((R**2 + z**2)/np.abs((a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4))))))/(np.pi*(R**2 + z**2)**2.5))


@jit(nopython=True)
def nu(rho, z, M, MD, b, m):  # Schwarzschild \nu potential: nuSch
    """Total nu metric potential: Schwarzschild BH, Morgan-Morgan disk, or their sum.

    Args:
        rho, z: Weyl cylindrical coordinates.
        M: black-hole mass parameter.
        MD: disk mass parameter.
        b: disk radius parameter (passed through to nuD as `a`).
        m: component selector -- 0: BH only (nuSch), 1: disk only (nuD), 2: BH + disk.

    Returns:
        The selected nu potential value.
    """
    # BH
    # DISK

    nuSch = math.log((d1(rho, z, M) + d2(rho, z, M) - 2*M) / (d1(rho, z, M) + d2(rho, z, M) + 2*M))

    if m == 0:

        return nuSch

    elif m == 1:


        return nuD(rho, z, MD, b)

    elif m == 2:

        return (nuSch + nuD(rho, z, MD, b))


@jit(nopython=True)
def lambSch(rho, z, M, MD, b):  # Schwarzshild \lambda potential: lambSch
    """Closed-form Schwarzschild lambda potential (integration boundary value for lamb_Mat)."""
    sigma = np.sqrt((rho**2 + z**2 + M**2)**2 - 4*z**2*M**2)
    return np.log(((d1(rho, z, M) + d2(rho, z, M))**2 - 4*M**2) / (4*sigma))


#####################################################################

########      TO CALCULATE THE INITIAL OBSERVER'S RHO       #########

def d1_i(rho, z, M):
    """Observer-frame (non-jitted) copy of d1, used only to locate the initial observer position via fsolve."""
    return np.sqrt(rho**2 + (z - M)**2)


def d2_i(rho, z, M):
    """Observer-frame (non-jitted) copy of d2."""
    return np.sqrt(rho**2 + (z + M)**2)


def xi2_i(R, z, a):
    """Observer-frame (non-jitted) copy of xi2."""
    return (np.sqrt((a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)) / (R**2 + z**2)) / np.sqrt(2))


def nuD_i(R, z, M, a):
    """Observer-frame (non-jitted) copy of nuD."""
    # if (np.abs(z) < 10**-6 and R >= a):
    if xi2_i(R, z, a) == 0.0:
        # print("a")
        return ((M*np.sqrt(np.abs(a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))*(-3*a**2 + R**2 + z**2 + 3*np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))/(np.sqrt(2)*np.pi*(R**2 + z**2)**2) - (2*M*(-(a**2*(R**2 - 2*z**2)) + 2*(R**2 + z**2)**2)*np.pi/2)/(np.pi*(R**2 + z**2)**2.5))
    # if (xi2(R,z,a) == 0.0):
    #     return  (- (2*M*(-(a**2*(R**2 - 2*z**2)) + 2*(R**2 + z**2)**2)*np.pi/2)/(np.pi*(R**2 + z**2)**2.5) )
    else:
        # print("b")
        return ((M*np.sqrt(np.abs(a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))*(-3*a**2 + R**2 + z**2 + 3*np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))/(np.sqrt(2)*np.pi*(R**2 + z**2)**2) - (2*M*(-(a**2*(R**2 - 2*z**2)) + 2*(R**2 + z**2)**2)*np.arctan(np.sqrt(2)*np.sqrt((R**2 + z**2)/np.abs((a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4))))))/(np.pi*(R**2 + z**2)**2.5))


def nu_i(rho, z, M, MD, b, m):  # Schwarzschild \nu potential: nuSch
    """Observer-frame (non-jitted) copy of nu, used only to locate the initial observer position via fsolve.

    #NOTE: the m == 1 branch calls the jitted `nuD`, not `nuD_i` -- kept as
    in the original scripts (a pre-existing quirk, not introduced here).
    """
    # BH
    # DISK

    nuSch = math.log((d1_i(rho, z, M) + d2_i(rho, z, M) - 2*M) / (d1_i(rho, z, M) + d2_i(rho, z, M) + 2*M))

    if m == 0:

        return nuSch

    elif m == 1:


        return nuD(rho, z, MD, b)

    elif m == 2:

        return (nuSch + nuD_i(rho, z, MD, b))


def gpp_i(rho, z, M, MD, b):
    """Observer-frame (non-jitted) copy of gpp, used to solve for the initial observer's rho via fsolve."""
    return rho**2 * math.exp(- nu_i(rho, z, M, MD, b, 2))


#####################################################################
#####################################################################
#####################################################################


@jit(nopython=True)
def derNU(rho, z, M, MD, b, l, m, hder):
    """Finite-difference derivative of `nu` with respect to rho (l=0) or z (l=1), component m."""
    return derivative(nu, l, rho, z, hder, M, MD, b, m)


@jit(nopython=True)
def dlamb(z, rho, M, MD, b, l, m, hder):
    """Integrand for lambda's z-derivative (l=1) or rho-derivative (l=0), from the vacuum Weyl field equations.

    #NOTE: argument order is (z, rho, ...) here (z first) so that `sci.quad`
    can integrate over z while treating rho as a fixed parameter in `lamb_Mat`.
    """
    if l == 0:
        return 0.5 * rho * (derNU(rho, z, M, MD, b, 0, m, hder)**2 - derNU(rho, z, M, MD, b, 1, m, hder)**2)

    elif l == 1:
        return rho * derNU(rho, z, M, MD, b, 0, m, hder) * derNU(rho, z, M, MD, b, 1, m, hder)


@jit(nopython=True)
def dlamb2(rho, z, M, MD, b, l, m, hder):
    """Same integrand as `dlamb`, but with (rho, z) argument order so `sci.quad` can integrate over rho."""
    if l == 0:
        return 0.5 * rho * (derNU(rho, z, M, MD, b, 0, m, hder)**2 - derNU(rho, z, M, MD, b, 1, m, hder)**2)

    elif l == 1:
        return rho * derNU(rho, z, M, MD, b, 0, m, hder) * derNU(rho, z, M, MD, b, 1, m, hder)


@jit(nopython=True)
def lamb(rho, z, M, MD, b, m):
    """Bilinear interpolation of lambda(rho, z) from the preloaded matrix `Mat_nu`.

    `Mat_nu` (set via `load_matrix`, from a matrix produced by
    generate_matriz.py's `lamb_Mat` quadrature) is indexed on a zref=Rref=40
    grid; (i0, j0) are the integer grid indices below the target point and
    (I, J) are the fractional interpolation weights within that grid cell.

    Args:
        rho, z: target Weyl coordinates.
        M, MD, b, m: unused here except implicitly via the preloaded matrix
            (kept in the signature for interface parity with `nu`/`lamb_Mat`).

    Returns:
        Bilinearly-interpolated lambda(rho, z).
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

    return ((f00 + (f01 - f00)*J) + (f10 - f00 + (f11 + f00 - f01 - f10)*J) * I)


# In[32]:


# Metric components (BH + disk, m=2)
@jit(nopython=True)
def gtt(rho, z, M, MD, b):
    """g_tt metric component at (rho, z)."""
    return -math.exp(nu(rho, z, M, MD, b, 2))


@jit(nopython=True)
def grr(rho, z, M, MD, b):
    """g_rho,rho metric component at (rho, z)."""
    return math.exp(lamb(rho, z, M, MD, b, 2) - nu(rho, z, M, MD, b, 2))


@jit(nopython=True)
def gzz(rho, z, M, MD, b):
    """g_zz metric component at (rho, z)."""
    return math.exp(lamb(rho, z, M, MD, b, 2) - nu(rho, z, M, MD, b, 2))


@jit(nopython=True)
def gpp(rho, z, M, MD, b):
    """g_phi,phi metric component at (rho, z)."""
    return rho**2 * math.exp(- nu(rho, z, M, MD, b, 2))

##########################


@jit(nopython=True)
def zeta(rho, z, M, MD, b):
    """Redshift-like normalization factor, sqrt(-1/g_tt), at (rho, z)."""
    return np.sqrt(-1 / gtt(rho, z, M, MD, b))

###############


# Momenta & initial velocities, from photon emission angles (alfa, beta)
@jit(nopython=True)
def dthe(rho, z, M, MD, b, alfa):
    """Initial d(z)/d(affine parameter)-like "theta" velocity component from emission angle alfa."""
    return 1 / np.sqrt(gzz(rho, z, M, MD, b)) * np.sin(alfa)


@jit(nopython=True)
def dr(rho, z, M, MD, b, alfa, beta):
    """Initial drho/d(affine parameter) velocity component from emission angles alfa, beta."""
    return 1 / np.sqrt(grr(rho, z, M, MD, b)) * np.cos(alfa) * np.cos(beta)


@jit(nopython=True)
def Pphi(rho, z, M, MD, b, alfa, beta):
    """Conserved photon angular momentum from local emission angles (alfa, beta)."""
    return np.sqrt(gpp(rho, z, M, MD, b)) * np.sin(beta) * np.cos(alfa)


@jit(nopython=True)
def Pt(rho, z, M, MD, b, alfa, beta):
    """Conserved photon energy at the initial point."""
    return -1 / zeta(rho, z, M, MD, b)


@jit(nopython=True)
def dphi(rho, z, M, MD, b, alfa, beta, p_phi):
    """d(phi)/d(affine parameter) from the conserved angular momentum p_phi."""
    return 1 / gpp(rho, z, M, MD, b) * p_phi


@jit(nopython=True)
def dt(rho, z, M, MD, b, alfa, beta, p_t):
    """dt/d(affine parameter) from the conserved energy p_t."""
    return 1 / gtt(rho, z, M, MD, b) * p_t


# In[33]:


@jit(nopython=True, parallel=True)
def run_kut4_mod(F, x, y, h, *args):
    """Adaptive-step embedded Runge-Kutta-Fehlberg 4(5) (RKF45) integrator step.

    Advances the ODE system y' = F(x, y, *args) by one step, estimating
    local error from the 4th- and 5th-order solutions (y4, y5) and shrinking
    `h` (bounded by hmin/hmax) until the error tolerance `tol` is met.

    Args:
        F: right-hand-side function, called as F(x, y, *args).
        x: current independent variable.
        y: current state vector.
        h: current (signed) step size.
        *args: extra positional arguments forwarded to F.

    Returns:
        (h, x, y): the (possibly reduced) step size actually used, and the
        new (x, y) after advancing by that step.
    """
    hmax = -0.04
    hmin = -10**-7
    tol = 10**-4

    K1 = h * F(x, y, *args)
    K2 = h * F(x + h/4, y + K1/4, *args)
    K3 = h * F(x + 3/8*h, y + 3/32*K1 + 9/32*K2, *args)
    K4 = h * F(x + 12/13*h, y + 1932/2197*K1 - 7200/2197*K2 + 7296/2197*K3, *args)
    K5 = h * F(x + h, y + 439/216*K1 - 8*K2 + 3680/513*K3 - 845/4104*K4, *args)
    K6 = h * F(x + h/2, y - 8/27*K1 + 2*K2 - 3544/2565*K3 + 1859/4104*K4 - 11/40*K5, *args)

    y4 = y + 25/216*K1 + 1408/2565*K3 + 2197/4101*K4 - K5/5
    y5 = y + 16/135*K1 + 6656/12825*K3 + 28561/56430*K4 - 9/50*K5 + 2/55*K6

    error = np.linalg.norm(y5 - y4)
    delta = pow(1.0/2.0, (1.0/4.0)) * pow(tol / error, (1.0/4.0))

    if error < tol:
        if np.abs(delta*h) < np.abs(hmin):
            h = hmin
        elif np.abs(delta*h) > np.abs(hmax):
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
            K2 = h * F(x + h/4, y + K1/4, *args)
            K3 = h * F(x + 3/8*h, y + 3/32*K1 + 9/32*K2, *args)
            K4 = h * F(x + 12/13*h, y + 1932/2197*K1 - 7200/2197*K2 + 7296/2197*K3, *args)
            K5 = h * F(x + h, y + 439/216*K1 - 8*K2 + 3680/513*K3 - 845/4104*K4, *args)
            K6 = h * F(x + h/2, y - 8/27*K1 + 2*K2 - 3544/2565*K3 + 1859/4104*K4 - 11/40*K5, *args)

            y4 = y + 25/216*K1 + 1408/2565*K3 + 2197/4101*K4 - K5/5
            y5 = y + 16/135*K1 + 6656/12825*K3 + 28561/56430*K4 - 9/50*K5 + 2/55*K6

            error = np.linalg.norm(y5 - y4)
            delta = pow(1.0/2.0, (1.0/4.0)) * pow(tol / error, (1.0/4.0))

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
