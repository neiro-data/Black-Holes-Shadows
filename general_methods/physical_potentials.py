#!/usr/bin/env python
# coding: utf-8
"""Metric potentials (nu, nuD, lambSch) and their finite-difference derivatives
(derNU, dlamb, dlamb2) for the Schwarzschild BH + Morgan-Morgan disk spacetime,
plus the non-jitted "_i" observer-frame copies used only to locate the initial
observer position via fsolve.

Split out of the former weyl_core.py -- see general_methods/__init__.py.
"""

import math
import numpy as np
from numba import jit

from .mathematical_formulas import derivative
from .physical_quantities import d1, d2, xi2, d1_i, d2_i, xi2_i

__all__ = [
    "nuD", "nu", "lambSch", "derNU", "dlamb", "dlamb2",
    "nuD_i", "nu_i",
]


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
    if xi2(R, z, a) == 0.0:
        return ((M*np.sqrt(np.abs(a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))*(-3*a**2 + R**2 + z**2 + 3*np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))/(np.sqrt(2)*np.pi*(R**2 + z**2)**2) - (2*M*(-(a**2*(R**2 - 2*z**2)) + 2*(R**2 + z**2)**2)*np.pi/2)/(np.pi*(R**2 + z**2)**2.5))
    else:
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
# Non-jitted copies, used only to locate the initial observer position via
# fsolve. Kept intentionally in parallel with `nu`/`nuD` above.

def nuD_i(R, z, M, a):
    """Observer-frame (non-jitted) copy of nuD."""
    if xi2_i(R, z, a) == 0.0:
        return ((M*np.sqrt(np.abs(a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))*(-3*a**2 + R**2 + z**2 + 3*np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))/(np.sqrt(2)*np.pi*(R**2 + z**2)**2) - (2*M*(-(a**2*(R**2 - 2*z**2)) + 2*(R**2 + z**2)**2)*np.pi/2)/(np.pi*(R**2 + z**2)**2.5))
    else:
        return ((M*np.sqrt(np.abs(a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))*(-3*a**2 + R**2 + z**2 + 3*np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))/(np.sqrt(2)*np.pi*(R**2 + z**2)**2) - (2*M*(-(a**2*(R**2 - 2*z**2)) + 2*(R**2 + z**2)**2)*np.arctan(np.sqrt(2)*np.sqrt((R**2 + z**2)/np.abs((a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4))))))/(np.pi*(R**2 + z**2)**2.5))


def nu_i(rho, z, M, MD, b, m):  # Schwarzschild \nu potential: nuSch
    """Observer-frame (non-jitted) copy of nu, used only to locate the initial observer position via fsolve.

    NOTE: the m == 1 branch calls the jitted `nuD`, not `nuD_i` -- kept as
    in the original weyl_core.py (a pre-existing quirk, not introduced here).
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


#####################################################################
#####################################################################


@jit(nopython=True)
def derNU(rho, z, M, MD, b, l, m, hder):
    """Finite-difference derivative of `nu` with respect to rho (l=0) or z (l=1), component m."""
    return derivative(nu, l, rho, z, hder, M, MD, b, m)


@jit(nopython=True)
def dlamb(z, rho, M, MD, b, l, m, hder):
    """Integrand for lambda's z-derivative (l=1) or rho-derivative (l=0), from the vacuum Weyl field equations.

    NOTE: argument order is (z, rho, ...) here (z first) so that `sci.quad`
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
