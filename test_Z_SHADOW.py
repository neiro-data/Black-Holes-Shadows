#!/usr/bin/env python
# coding: utf-8
"""Serial (non-parallel) shadow-grid generator for the Schwarzschild BH +
Morgan-Morgan disk system, in Weyl coordinates.

Computes null-geodesic photon trajectories via a serial (plain `range`, no
numba `prange`) double loop over emission angles, and classifies each pixel
by its final rho/z, tagging pixels beyond the disk plane with a +50.0 offset
on z (see `func`'s "Pontos do Disco" branch) so the disk can be told apart
from the black hole in post-processing. Depends on a pre-tabulated
lambda-potential matrix loaded via `general_methods.load_matrix("Mat_nu_disk0.0")`
(produced by generate_matriz.py). Unlike its siblings, this file's np.savetxt
calls for Mat/Mz are commented out (see the driver section) -- it appears to
have been used as an interactive/exploratory run rather than a batch
data-producing one, with an inline matplotlib figure shown instead.

The shared physics core (simps, derivative, d1, d2, xi2, nuD, nu, lambSch,
gpp/gtt/grr/gzz, zeta, dthe, dr, Pphi, Pt, dphi, dt, the `_i` observer-frame
variants, derNU, dlamb, dlamb2, lamb, run_kut4_mod) now lives in general_methods
and is imported below -- see general_methods's module docstring for the
canonicalization notes (xi2's np.abs guard, unified decorators). `geo` and
`func` remain local to this file: `func` here keeps a *live* disk-crossing
classification branch (the `+50.0` z tag) that the basis test_parallel_SHADOW.py
carries only as dead/commented code -- this is this script's distinguishing
variant behaviour and the reason it was kept as its own file rather than
folded into the basis (see claude_interaction_steps.md, Interaction 3).
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import time
from numba import jit
from scipy.optimize import fsolve

import general_methods
from general_methods import *


@jit(nopython=True)
def geo(t, z, M, alfa, beta, rho0, z0, MD, b, hder):
    """Geodesic equations of motion (right-hand side), via the Weyl nu/lambda potentials and their derivatives.

    Args:
        t: current affine parameter (unused, kept for the RKF45 interface).
        z: state vector [rho, drho/dtau, z_coord, dz/dtau].
        M, MD, b: BH mass, disk mass, disk radius.
        alfa, beta: photon emission angles at the initial point.
        rho0, z0: initial emission point.
        hder: finite-difference step used inside derNU.

    Returns:
        np.array([drho, d2rho, dz, d2z]).
    """
    pphi = Pphi(rho0, z0, M, MD, b, alfa, beta)
    pt = Pt(rho0, z0, M, MD, b, alfa, beta)

    d2Rdt = -(0.5*math.exp(2*nu(z[0], z[2], M, MD, b, 2) - lamb(z[0], z[2], M, MD, b, 2))*derNU(z[0], z[2], M, MD, b, 0, 2, hder) * dt(z[0], z[2], M, MD, b, alfa, beta, pt)**2 + 0.5*(dlamb2(z[0], z[2], M, MD, b, 0, 2, hder) - derNU(z[0], z[2], M, MD, b, 0, 2, hder))*z[1]**2 + (dlamb2(z[0], z[2], M, MD, b, 1, 2, hder) - derNU(z[0], z[2], M, MD, b, 1, 2, hder))*z[1]*z[3] - 0.5*(dlamb2(z[0], z[2], M, MD, b, 0, 2, hder) - derNU(z[0], z[2], M, MD, b, 0, 2, hder))*z[3]**2 + 0.5*math.exp(-lamb(z[0], z[2], M, MD, b, 2))*z[0]*(-2 + z[0]*derNU(z[0], z[2], M, MD, b, 0, 2, hder))*dphi(z[0], z[2], M, MD, b, alfa, beta, pphi)**2)

    d2Thedt = -(0.5*math.exp(2*nu(z[0], z[2], M, MD, b, 2) - lamb(z[0], z[2], M, MD, b, 2))*derNU(z[0], z[2], M, MD, b, 1, 2, hder) * dt(z[0], z[2], M, MD, b, alfa, beta, pt)**2 - 0.5*(dlamb2(z[0], z[2], M, MD, b, 1, 2, hder) - derNU(z[0], z[2], M, MD, b, 1, 2, hder))*z[1]**2 + (dlamb2(z[0], z[2], M, MD, b, 0, 2, hder) - derNU(z[0], z[2], M, MD, b, 0, 2, hder))*z[1]*z[3] + 0.5*(dlamb2(z[0], z[2], M, MD, b, 1, 2, hder) - derNU(z[0], z[2], M, MD, b, 1, 2, hder))*z[3]**2 + 0.5*math.exp(-lamb(z[0], z[2], M, MD, b, 2))*z[0]**2*derNU(z[0], z[2], M, MD, b, 1, 2, hder)*dphi(z[0], z[2], M, MD, b, alfa, beta, pphi)**2)




    return np.array([z[1], d2Rdt, z[3], d2Thedt])


@jit(nopython=True)
def func(y, x, h, alfa, beta, M, rho0, z0, MD, b, hder):
    """Single-ray tracer: integrates one photon's geodesic until escape, capture, or crossing the disk plane.

    Repeatedly advances the state with `run_kut4_mod` while the photon's nu
    potential stays above -3.0 and its areal radius stays below 30.

    Args:
        y: initial state [rho0, drho, z0, dz].
        x: initial affine parameter.
        h: initial (negative) step size.
        alfa, beta: emission angles.
        M, rho0, z0, MD, b, hder: BH mass, initial emission point, disk
            mass, disk radius, finite-difference step.

    Returns:
        [rho, z] at the escape point, [0.001, 0.001] if captured, or
        [rho, z+50.0] if the ray crosses the disk plane (rho > b with a
        sign flip in z between consecutive steps) -- the +50.0 offset flags
        disk-crossing pixels for the classification pass in the driver below.

    #NOTE: `yf` is only ever assigned inside the while loop body; if the
    loop condition is already false on the first check, `return (yf)` would
    reference an unbound variable. Not fixed here per instructions.
    """
    Y = np.zeros((1, 4))

    Y = np.concatenate((Y, y.reshape((1, 4))), axis=0)

    while (nu(y[0], y[2], M, MD, b, 2) > -3.0 and np.sqrt(gpp(np.sqrt(y[0]**2 + y[2]**2), 0, M, MD, b)) < 30.0):

        (h, x, y) = run_kut4_mod(geo, x, y, h, M, alfa, beta, rho0, z0, MD, b, hder)

        Y = np.concatenate((Y, y.reshape((1, 4))), axis=0)

        if np.sqrt(gpp(np.sqrt(y[0]**2 + y[2]**2), 0, M, MD, b)) >= 30.0:
            yf = [Y[-1][0], Y[-1][2]]
            break

        ####   Pontos que caem no buraco negro    #####

        elif nu(y[0], y[2], M, MD, b, 2) <= -3.0:
            yf = [0.001, 0.001]
            break

        ####   Pontos do Disco    #####

        elif (y[0] > b and np.sign(Y[-1][2]) == - np.sign(Y[-2][2])):
            yf = [Y[-1][0], Y[-1][2] + 50.0]
            break

    return (yf)




# Driver: solve for the initial observer's rho, build a small (quarter-image)
# emission-angle grid, ray-trace it serially, classify each pixel, and show
# (rather than save) the resulting shadow figure -- an interactive/
# exploratory run, unlike the parallel production driver in
# test_parallel_SHADOW.py.
general_methods.load_matrix("Mat_nu_disk0.0")

start = time.time()
M = 1.0
MD = 0.0
z0 = 0.0
b = 6.0
hder = 10**-6
func_initial = lambda R: np.sqrt(gpp_i(R, z0, M, MD, b)) - 15.0

R_initial_guess = 10.0
R_solution = fsolve(func_initial, R_initial_guess)

rho0 = float(R_solution)
print(rho0)




alfaa = np.linspace(-np.arctan(10/15), np.arctan(10/15), 80)
betaa = np.linspace(np.arctan(10/15), -np.arctan(10/15), 80)

alfa = np.linspace(alfaa[0], alfaa[int(len(alfaa)/2) - 1], int(len(alfaa)/2))
beta = np.linspace(betaa[0], betaa[int(len(betaa)/2) - 1], int(len(betaa)/2))


Mat = np.zeros((len(alfa), len(beta)))
Mz = np.zeros((len(alfa), len(beta)))

for i in range(len(alfa)):
    for j in range(len(beta)):

        y = np.array([rho0, dr(rho0, z0, M, MD, b, alfa[i], beta[j]), z0, dthe(rho0, z0, M, MD, b, alfa[i])])
        (Mat[i, j], Mz[i, j]) = func(y, 300.0, -0.02, alfa[i], beta[j], M, rho0, z0, MD, b, hder)

end = time.time()
print(end - start)


# np.savetxt('Mat',Mat)
# np.savetxt('Mz',Mz)






M2 = np.zeros((len(Mat), len(Mat[0])))
Mat2 = np.zeros((len(alfa), len(beta)))
Mz2 = np.zeros((len(alfa), len(beta)))

# Live classification: Mz/Mat values above 49.0/50.0 are unshifted (the
# +50.0 disk-crossing offset set by func() above); a pixel is "captured"
# (M2=1) if its unshifted radius is <= 0.002, or "beyond the disk" (M2=2) if
# radius > b and z escaped past the wraparound; otherwise "neither" (M2=0).
for i in range(len(Mz)):
    for j in range(len(Mz[0])):
        if Mz[i, j] > 49.0:
            Mz2[i, j] = Mz[i, j] - 50.0
        else:
            Mz2[i, j] = Mz[i, j]

for i in range(len(Mat)):
    for j in range(len(Mat[0])):
        if Mat[i, j] > 50.0:
            Mat2[i, j] = Mat[i, j] - 50.0
        else:
            Mat2[i, j] = Mat[i, j]
count1 = 0
count2 = 0
for i in range(len(Mat)):
    for j in range(len(Mat[0])):

        if (Mat2[i, j] <= 0.002):  # and Mz[i,j] < M):
            M2[i, j] = 1
            count1 += 1

        elif (Mat2[i, j] > b and Mz[i, j] > 49.0):
            M2[i, j] = 2


        else:
            M2[i, j] = 0
print(count1, count2)


plt.figure(figsize=(15, 15))


# c_map = [[1,1,1],[0, 0, 0], [0.192, 0.192, 0.192],[1,0,0]]
c_map = [[1, 1, 1], [0, 0, 0], [0.192, 0.192, 0.192]]
cm = matplotlib.colors.ListedColormap(c_map)

plt.imshow(M2, cmap=cm, extent=[-beta[0], -beta[-1], alfa[0], alfa[-1]])
# plt.colorbar()


plt.xlabel("$\\beta$", size=30)
plt.ylabel("$\\alpha$", size=30)

plt.tick_params(axis='both', which='major', labelsize=14)
# plt.legend(loc=10, bbox_to_anchor=(0.85, 0.9), ncol=1,fontsize=17)

plt.show()
# plt.savefig("Schwarzschild_Weyl_coords_154x154")
