#!/usr/bin/env python
# coding: utf-8
"""Weyl-coordinate distance quantities used by the Schwarzschild BH + Morgan-Morgan
disk potentials: distances to the BH rod endpoints (d1, d2) and the disk
oblate-spheroidal-like coordinate (xi2), plus their non-jitted "_i" observer-frame
copies used only to locate the initial observer position via fsolve.

Split out of the former weyl_core.py -- see general_methods/__init__.py.
"""

import numpy as np
from numba import jit

__all__ = ["d1", "d2", "xi2", "d1_i", "d2_i", "xi2_i"]


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

    Canonical (np.abs-guarded) form: matches how the lookup table was
    actually built by generate_matriz.py, and avoids NaN for points where
    the radicand is a tiny negative number near the disk edge.
    """
    return (np.sqrt(np.abs((a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)) / (R**2 + z**2))) / np.sqrt(2))


#####################################################################
########      TO CALCULATE THE INITIAL OBSERVER'S RHO       #########
# Non-jitted copies, used only to locate the initial observer position via
# fsolve. Kept intentionally, in parallel with the jitted versions above --
# see general_methods/spacetime_metrics.py's `gpp_i` (the only one of this
# chain called directly by the consumer ray-tracer scripts).

def d1_i(rho, z, M):
    """Observer-frame (non-jitted) copy of d1, used only to locate the initial observer position via fsolve."""
    return np.sqrt(rho**2 + (z - M)**2)


def d2_i(rho, z, M):
    """Observer-frame (non-jitted) copy of d2."""
    return np.sqrt(rho**2 + (z + M)**2)


def xi2_i(R, z, a):
    """Observer-frame (non-jitted) copy of xi2."""
    return (np.sqrt((a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)) / (R**2 + z**2)) / np.sqrt(2))
