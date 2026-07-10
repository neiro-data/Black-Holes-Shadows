#!/usr/bin/env python
# coding: utf-8
"""Metric components, momenta, and initial-velocity components for the
Schwarzschild BH + Morgan-Morgan disk spacetime, plus the non-jitted `gpp_i`
observer-frame copy used only to locate the initial observer position via
fsolve.

All of the functions below remain module-level (jitted where the original
was jitted) because several of them (`gpp`) are called by bare name inside
other scripts' own `@jit(nopython=True)` geodesic code (test_parallel_SHADOW.py,
test_Z_SHADOW.py, test_symmetry_lensing.py) -- numba nopython mode can't call
arbitrary Python instance methods, so wrapping them directly in a class would
break jit compilation there. `Metric` and `MetricDerivatives` below are thin,
non-jitted convenience wrappers on top of these free functions for callers
who want an object-oriented interface; they don't replace the free functions.

Split out of the former weyl_core.py -- see general_methods/__init__.py.
"""

import math
import numpy as np
from numba import jit

from .physical_potentials import nu, nu_i
from .mathematical_formulas import lamb

__all__ = [
    "gtt", "grr", "gzz", "gpp", "gpp_i", "zeta",
    "dthe", "dr", "Pphi", "Pt", "dphi", "dt",
    "Metric", "MetricDerivatives",
]


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


#####################################################################
########      TO CALCULATE THE INITIAL OBSERVER'S RHO       #########
# Non-jitted copy, used only to locate the initial observer position via
# fsolve. Kept intentionally, in parallel with `gpp` above.

def gpp_i(rho, z, M, MD, b):
    """Observer-frame (non-jitted) copy of gpp, used to solve for the initial observer's rho via fsolve."""
    return rho**2 * math.exp(- nu_i(rho, z, M, MD, b, 2))


#####################################################################


@jit(nopython=True)
def zeta(rho, z, M, MD, b):
    """Redshift-like normalization factor, sqrt(-1/g_tt), at (rho, z)."""
    return np.sqrt(-1 / gtt(rho, z, M, MD, b))


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


class Metric:
    """Object-oriented convenience wrapper around the module-level metric
    functions above. Bound to fixed (M, MD, b) black-hole/disk parameters;
    methods take just (rho, z, ...) and delegate to the free jitted functions.

    This class is not used by the numba-jitted `geo`/`func` code in the
    consumer ray-tracer scripts (see module docstring) -- it's a plain-Python
    interface for exploratory/analysis code.
    """

    def __init__(self, M, MD, b):
        self.M = M
        self.MD = MD
        self.b = b

    def gtt(self, rho, z):
        return gtt(rho, z, self.M, self.MD, self.b)

    def grr(self, rho, z):
        return grr(rho, z, self.M, self.MD, self.b)

    def gzz(self, rho, z):
        return gzz(rho, z, self.M, self.MD, self.b)

    def gpp(self, rho, z):
        return gpp(rho, z, self.M, self.MD, self.b)

    def zeta(self, rho, z):
        return zeta(rho, z, self.M, self.MD, self.b)

    def Pphi(self, rho, z, alfa, beta):
        return Pphi(rho, z, self.M, self.MD, self.b, alfa, beta)

    def Pt(self, rho, z, alfa, beta):
        return Pt(rho, z, self.M, self.MD, self.b, alfa, beta)


class MetricDerivatives:
    """Object-oriented convenience wrapper around the initial-velocity/momentum
    rate functions above (dthe, dr, dphi, dt). Bound to fixed (M, MD, b)
    black-hole/disk parameters; see `Metric` docstring for the same caveat
    about numba jit compatibility.
    """

    def __init__(self, M, MD, b):
        self.M = M
        self.MD = MD
        self.b = b

    def dthe(self, rho, z, alfa):
        return dthe(rho, z, self.M, self.MD, self.b, alfa)

    def dr(self, rho, z, alfa, beta):
        return dr(rho, z, self.M, self.MD, self.b, alfa, beta)

    def dphi(self, rho, z, alfa, beta, p_phi):
        return dphi(rho, z, self.M, self.MD, self.b, alfa, beta, p_phi)

    def dt(self, rho, z, alfa, beta, p_t):
        return dt(rho, z, self.M, self.MD, self.b, alfa, beta, p_t)
