#!/usr/bin/env python
# coding: utf-8
"""Numerical utilities shared by the Weyl-family ray tracers: Simpson's-rule
integration, finite-difference derivatives, the RKF45 adaptive integrator, and
bilinear interpolation of the pre-tabulated lambda-potential matrix (`lamb`).

Split out of the former weyl_core.py -- see general_methods/__init__.py.
"""

import math
import numpy as np
from numba import jit

__all__ = [
    "simps", "derivative", "run_kut4_mod", "lamb", "load_matrix",
]

# forceobj=True: this does file I/O (np.loadtxt), which numba cannot compile in
# nopython mode; modern numba (>=0.59) no longer auto-falls back to object mode
# for nopython=False, so object mode must be requested explicitly.
#@jit(forceobj=True)
def load_matrix(path):
    """Load and return the pre-tabulated lambda-potential matrix (produced by
    generate_matriz.py). The returned array must be passed explicitly as the
    `Mat_nu` argument to `lamb`/grr/gzz/dr/dthe and each script's geo/func --
    it is no longer stored in a module-level global (numba freezes nopython
    globals as compile-time constants, so a reloadable global cannot work)."""
    return np.loadtxt(path)


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


@jit(nopython=True)
def lamb(rho, z, M, MD, b, m, Mat_nu):
    """Bilinear interpolation of lambda(rho, z) from the lambda matrix `Mat_nu`.

    `Mat_nu` (obtained from `load_matrix`, from a matrix produced by
    generate_matriz.py's `lamb_Mat` quadrature) is passed in explicitly and
    indexed on a zref=Rref=40 grid; (i0, j0) are the integer grid indices below
    the target point and (I, J) are the fractional interpolation weights within
    that grid cell.

    Args:
        rho, z: target Weyl coordinates.
        M, MD, b, m: unused here except implicitly via `Mat_nu` (kept in the
            signature for interface parity with `nu`/`lamb_Mat`).
        Mat_nu: the pre-tabulated lambda-potential matrix (from `load_matrix`).

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

    NOTE: kept `parallel=True` (rather than plain nopython) to match how it
    was originally paired with test_parallel_SHADOW.py's other parallel=True
    functions -- calling a parallel-decorated `geo` through a plain-nopython
    `run_kut4_mod` triggers numba's workqueue threading layer to abort with
    "Concurrent access has been detected". See the former weyl_core.py module
    docstring (preserved in general_methods/__init__.py) for full details.
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
