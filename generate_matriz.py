#!/usr/bin/env python
# coding: utf-8
"""Precompute the non-linear lambda (\\lambda) metric potential matrix for the
Weyl-coordinate Schwarzschild-BH + Morgan-Morgan-disk spacetime used by the
shadow/lensing ray tracers in this repository (test_Z_SHADOW.py,
test2_Z_SHADOW.py, test_parallel_SHADOW.py, test_symmetry_lensing.py).

Those ray tracers need lambda(rho, z) at arbitrary (rho, z) along a photon's
path, but lambda is only known in closed form on the symmetry axis: it must
otherwise be obtained by numerically integrating its (rho, z) derivatives
(dlamb/dlamb2) starting from a reference point at "infinity" (here rho=z=40).
Doing that integral on every ray-tracing step would be prohibitively slow, so
this script tabulates lambda on a (z, rho) grid once, via `lamb_Mat`, and
saves it to disk. The ray tracers then bilinearly interpolate this saved
matrix at runtime (see `lamb()` in the sibling scripts) instead of
re-integrating.

Output: a plain-text matrix (`Mat_constA_Mbh_0.9` by default) consumed via
`np.loadtxt("Mat_nu_disk*")` by the ray-tracing scripts.

This is a straight indentation/documentation pass over the original
Jupyter-notebook export (note the `# In[NN]` cell markers, kept for
provenance). No behaviour was changed.

#Duplicated: `simps`, `derivative`, `d1`, `d2`, `xi2`, `nuD`, `nu`, `lambSch`,
`gpp`, the `_i` observer-frame variants, `derNU`, `dlamb`, `dlamb2` are
copy-pasted verbatim (module-name aside) across test_Z_SHADOW.py,
test2_Z_SHADOW.py, test_parallel_SHADOW.py and test_symmetry_lensing.py.
See test_parallel_SHADOW.py for the canonical/most complete copy.
"""

# In[25]:


import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import time
from numba import jit
import cmath
import scipy.integrate as sci
from scipy.optimize import fsolve



# In[26]:


##########################
# NUMERICAL Integration and Derivatives
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
    """Oblate-spheroidal-like coordinate xi used in the Morgan-Morgan disk potential nuD, for disk parameter a."""
    return (np.sqrt(np.abs((a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)) / (R**2 + z**2))) / np.sqrt(2))


#######################################################################################################


####################
# Potenciais    ####
####################


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
    """Closed-form Schwarzschild lambda potential (used as a reference/boundary value for the numeric lambda integration)."""
    sigma = np.sqrt((rho**2 + z**2 + M**2)**2 - 4*z**2*M**2)
    return np.log(((d1(rho, z, M) + d2(rho, z, M))**2 - 4*M**2) / (4*sigma))


@jit(nopython=True)
def gpp(rho, z, M, MD, b):
    """g_phiphi metric component (BH+disk nu potential, m=2) at (rho, z)."""
    return rho**2 * math.exp(- nu(rho, z, M, MD, b, 2))



#####################################################################

########      TO CALCULATE THE INITIAL OBSERVER'S RHO       #########

def d1_i(rho, z, M):
    """Observer-frame (non-jitted) copy of d1, used only to locate the initial observer position via fsolve."""
    return np.sqrt(rho**2 + (z - M)**2)


def d2_i(rho, z, M):
    """Observer-frame (non-jitted) copy of d2, used only to locate the initial observer position via fsolve."""
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

    #NOTE: branches m == 1 and m == 2 call the jitted `nuD`/module-level `nuD_i`
    inconsistently with the `_i` naming convention elsewhere in this function
    (m == 1 calls `nuD`, not `nuD_i`) -- kept as in the original.
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

###################################################################################
###################################################################################
###################################################################################



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
    """Same integrand as `dlamb`, but with (rho, z) argument order so `sci.quad` can integrate over rho.

    #Duplicated: identical body to `dlamb`, only the parameter order differs.
    """
    if l == 0:
        return 0.5 * rho * (derNU(rho, z, M, MD, b, 0, m, hder)**2 - derNU(rho, z, M, MD, b, 1, m, hder)**2)

    elif l == 1:
        return rho * derNU(rho, z, M, MD, b, 0, m, hder) * derNU(rho, z, M, MD, b, 1, m, hder)


@jit
def lamb_Mat(rho, z, M, MD, b, m, hder):
    """Numerically integrate lambda(rho, z) from the reference point (40, 40) [or (40, -40) if z < 0].

    This is the expensive, ground-truth evaluation of lambda that the whole
    script exists to tabulate once via `Mat_nu`/`Mat_constA_*`, so that the
    ray-tracing scripts can interpolate instead of re-integrating per step.

    Args:
        rho, z: target Weyl coordinates.
        M, MD, b: BH mass, disk mass, disk radius parameters.
        m: nu-potential component selector forwarded to dlamb/dlamb2/lambSch.
        hder: finite-difference step used inside derNU.

    Returns:
        lambda(rho, z), by quadrature of its z- and rho-derivatives plus the
        closed-form Schwarzschild boundary value `lambSch` at (40, +/-40).
    """
    limit = 0.0
    if z >= limit:
        return sci.quad(dlamb, 40.0, z, (rho, M, MD, b, 1, m, hder))[0] + sci.quad(dlamb2, 40.0, rho, (40.0, M, MD, b, 0, m, hder))[0] + lambSch(40.0, 40.0, M + MD, MD, b)

    elif z < -limit:
        return sci.quad(dlamb, -40, z, (rho, M, MD, b, 1, m, hder))[0] + sci.quad(dlamb2, 40.0, rho, (-40.0, M, MD, b, 0, m, hder))[0] + lambSch(40.0, -40.0, M + MD, MD, b)



# Legacy: earlier version of lamb_Mat that used a raw 1/r Newtonian falloff
# (-2*(M+MD)/sqrt(40^2+40^2)) instead of the closed-form lambSch boundary
# value above. Superseded, kept for reference.
"""
@jit
def lamb_Mat(rho,z,M,MD,b,m,hder):
    limit = 0.0
    if z >= limit:
        return sci.quad(dlamb,40.0,z,(rho,M,MD,b,1,m,hder))[0] + sci.quad(dlamb2,40.0,rho,(40.0,M,MD,b,0,m,hder))[0] -2*(M+MD)/np.sqrt(40.0**2+40.0**2)

    elif z < -limit:
        return sci.quad(dlamb,-40,z,(rho,M,MD,b,1,m,hder))[0] + sci.quad(dlamb2,40.0,rho,(-40.0,M,MD,b,0,m,hder))[0] -2*(M+MD)/np.sqrt(40.0**2+40.0**2)
"""
# Legacy: yet another earlier lamb_Mat, single-branch (z >= 0 case only), commented out.
# @jit
# def lamb_Mat(rho,z,M,MD,b,m,hder):
#     return sci.quad(dlamb,40.0,z,(rho,M,MD,b,1,m,hder))[0] + sci.quad(dlamb2,40.0,rho,(40.0,M,MD,b,0,m,hder))[0] + lambSch(40.0,40.0,M+MD,MD,b)



########################################################################################

#######################################
#                                      #
# CREATE THE NON-LINEAR POTENTIAL MATRIX #
#                                      #
########################################

# Driver: tabulate lambda(rho, z) on a 1200x1200 grid spanning
# rho in [0, 40], z in [-40, 40], for a BH+disk configuration (M, MD, b below),
# and save it to disk for the ray tracers to interpolate.
z = np.linspace(40.0, -40.0, 1200)
rho = np.linspace(40.0, 0.0, 1200)
nu_Mat = np.zeros((len(z), len(rho)))


count = 0
M = 0.9
MD = 0.1
z0 = 0.0
b = 3.0
func_initial = lambda R: np.sqrt(gpp_i(R, z0, M, MD, b)) - 15.0

R_initial_guess = 10.0
R_solution = fsolve(func_initial, R_initial_guess)

rho0 = float(R_solution)

for i in range(len(z)):
    for j in range(len(rho)):
        nu_Mat[i, j] = lamb_Mat(rho[j], z[i], M, MD, b, 2, 10**-6)

# Replace any NaN entries (quadrature failures, typically near coordinate
# singularities) with a large negative sentinel so downstream interpolation
# doesn't propagate NaNs, and report how many grid points were affected.
for i in range(len(z)):
    for j in range(len(rho)):
        if np.isnan(nu_Mat[i, j]) == True:
            nu_Mat[i, j] = -20
            count += 1
            print(z[i], rho[j])

print(count)
# np.savetxt('Mat_nu_disk_big',nu_Mat)
np.savetxt('Mat_constA_Mbh_0.9', nu_Mat)


########################################################################################
