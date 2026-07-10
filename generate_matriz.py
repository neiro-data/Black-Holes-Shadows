#!/usr/bin/env python
# coding: utf-8
"""Precompute the non-linear lambda (\\lambda) metric potential matrix for the
Weyl-coordinate Schwarzschild-BH + Morgan-Morgan-disk spacetime used by the
shadow/lensing ray tracers in this repository (test_Z_SHADOW.py,
test_parallel_SHADOW.py, test_symmetry_lensing.py).

Those ray tracers need lambda(rho, z) at arbitrary (rho, z) along a photon's
path, but lambda is only known in closed form on the symmetry axis: it must
otherwise be obtained by numerically integrating its (rho, z) derivatives
(dlamb/dlamb2) starting from a reference point at "infinity" (here rho=z=40).
Doing that integral on every ray-tracing step would be prohibitively slow, so
this script tabulates lambda on a (z, rho) grid once, via `lamb_Mat`, and
saves it to disk. The ray tracers then bilinearly interpolate this saved
matrix at runtime (see `weyl_core.lamb()`) instead of re-integrating.

Output: a plain-text matrix (`Mat_constA_Mbh_0.9` by default) consumed via
`weyl_core.load_matrix("Mat_nu_disk*")` by the ray-tracing scripts.

This is a pipeline *base* script in the execution-order sense (it must run
before the ray tracers, since they load its output) -- distinct from
weyl_core.py, which is the comparison basis the shared helpers were
consolidated into. The shared physics core (simps, derivative, d1, d2, xi2,
nuD, nu, lambSch, gpp, the `_i` observer-frame variants, derNU, dlamb,
dlamb2) now lives in weyl_core.py and is imported below. `lamb_Mat` -- the
expensive quadrature this script exists to run -- and the tabulation driver
remain local: they are unique to this file among the family (see
claude_interaction_steps.md, Interaction 3 and 4).

This is a straight indentation/documentation pass over the original
Jupyter-notebook export (note the `# In[NN]` cell markers, kept for
provenance). No behaviour was changed beyond the extraction itself.
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

import weyl_core
from weyl_core import *


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
