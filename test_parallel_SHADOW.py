#!/usr/bin/env python
# coding: utf-8
"""Parallelized (numba `prange`) production shadow-grid generator for the
Schwarzschild BH + Morgan-Morgan disk system, in Weyl coordinates.

This is the most complete / production version of the black-hole shadow ray
tracer in this repo: it computes null-geodesic photon trajectories and uses
numba's `prange` (parallel range) inside `f_paral`, jitted with
`@jit(nopython=True, parallel=True)`, to fill the shadow grid across
multiple cores -- the only file in the family with true parallelism (the
serial siblings are test_Z_SHADOW.py, test_symmetry_lensing.py). It depends
on a pre-tabulated lambda-potential matrix loaded via
`weyl_core.load_matrix("Mat_nu_disk0.1")` (produced by generate_matriz.py),
and saves its output matrices (`Mat`, `Mz`) via np.savetxt for the plotting
scripts (symmetry.py, simetria_shadow*.py, lensing_image.py) to consume.

The shared physics core (simps, derivative, d1, d2, xi2, nuD, nu, lambSch,
gpp/gtt/grr/gzz, zeta, dthe, dr, Pphi, Pt, dphi, dt, the `_i` observer-frame
variants, derNU, dlamb, dlamb2, lamb, run_kut4_mod) now lives in weyl_core.py
and is imported below -- see weyl_core.py's module docstring for the
canonicalization notes (xi2's np.abs guard, unified decorators). `geo`,
`func`, `f_paral`, and the driver below remain local to this file: they are
this script's variant layer (parallel shadow classification), not shared.
"""

import math
import numpy as np
import time
from numba import jit
from numba import prange
from scipy.optimize import fsolve

import weyl_core
from weyl_core import *


@jit(nopython=True, parallel=True)
def geo(t, z, M, alfa, beta, rho0, z0, MD, b, hder):
    """Geodesic equations of motion (right-hand side), via the Weyl nu/lambda potentials and their derivatives.

    Args:
        t: current affine parameter (unused directly, kept for the RKF45 interface).
        z: state vector [rho, drho/dtau, z_coord, dz/dtau].
        M, MD, b: BH mass, disk mass, disk radius.
        alfa, beta: photon emission angles at the initial point, used (with
            rho0, z0) to compute the conserved momenta pphi, pt.
        rho0, z0: initial emission point (kept fixed since pphi, pt are
            constants of motion).
        hder: finite-difference step used inside derNU.

    Returns:
        np.array([drho, d2rho, dz, d2z]) -- the derivative of the state
        vector z, i.e. the 2nd-order radial/z photon equations derived from
        the Weyl metric potentials.
    """
    pphi = Pphi(rho0, z0, M, MD, b, alfa, beta)
    pt = Pt(rho0, z0, M, MD, b, alfa, beta)

    d2Rdt = -(0.5*math.exp(2*nu(z[0], z[2], M, MD, b, 2) - lamb(z[0], z[2], M, MD, b, 2))*derNU(z[0], z[2], M, MD, b, 0, 2, hder) * dt(z[0], z[2], M, MD, b, alfa, beta, pt)**2 + 0.5*(dlamb2(z[0], z[2], M, MD, b, 0, 2, hder) - derNU(z[0], z[2], M, MD, b, 0, 2, hder))*z[1]**2 + (dlamb2(z[0], z[2], M, MD, b, 1, 2, hder) - derNU(z[0], z[2], M, MD, b, 1, 2, hder))*z[1]*z[3] - 0.5*(dlamb2(z[0], z[2], M, MD, b, 0, 2, hder) - derNU(z[0], z[2], M, MD, b, 0, 2, hder))*z[3]**2 + 0.5*math.exp(-lamb(z[0], z[2], M, MD, b, 2))*z[0]*(-2 + z[0]*derNU(z[0], z[2], M, MD, b, 0, 2, hder))*dphi(z[0], z[2], M, MD, b, alfa, beta, pphi)**2)

    d2Thedt = -(0.5*math.exp(2*nu(z[0], z[2], M, MD, b, 2) - lamb(z[0], z[2], M, MD, b, 2))*derNU(z[0], z[2], M, MD, b, 1, 2, hder) * dt(z[0], z[2], M, MD, b, alfa, beta, pt)**2 - 0.5*(dlamb2(z[0], z[2], M, MD, b, 1, 2, hder) - derNU(z[0], z[2], M, MD, b, 1, 2, hder))*z[1]**2 + (dlamb2(z[0], z[2], M, MD, b, 0, 2, hder) - derNU(z[0], z[2], M, MD, b, 0, 2, hder))*z[1]*z[3] + 0.5*(dlamb2(z[0], z[2], M, MD, b, 1, 2, hder) - derNU(z[0], z[2], M, MD, b, 1, 2, hder))*z[3]**2 + 0.5*math.exp(-lamb(z[0], z[2], M, MD, b, 2))*z[0]**2*derNU(z[0], z[2], M, MD, b, 1, 2, hder)*dphi(z[0], z[2], M, MD, b, alfa, beta, pphi)**2)




    return np.array([z[1], d2Rdt, z[3], d2Thedt])


@jit(nopython=True, parallel=True)
def func(y, x, h, alfa, beta, M, rho0, z0, MD, b, hder):
    """Single-ray tracer: integrates one photon's geodesic until it escapes, is captured, or exits otherwise.

    Repeatedly advances the state with `run_kut4_mod` while the photon's nu
    potential stays above -3.0 and its areal radius stays below 30 (i.e.
    while it is neither captured nor escaped).

    Args:
        y: initial state [rho0, drho, z0, dz].
        x: initial affine parameter.
        h: initial (negative) step size.
        alfa, beta: emission angles.
        M, rho0, z0, MD, b, hder: BH mass, initial emission point, disk
            mass, disk radius, finite-difference step.

    Returns:
        [rho, z] at the escape point (areal radius >= 30), or the sentinel
        [0.001, 0.001] if the photon is captured (nu <= -3.0).

    #NOTE: `yf` is only ever assigned inside the while loop body (via the
    escape/capture branches); if the loop condition is already false on the
    very first check, `return (yf)` would reference an unbound variable.
    Not fixed here per instructions.
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

    return (yf)


@jit(nopython=True, parallel=True)
def f_paral(rho0, z0, M, MD, b, alfa, beta, hder):
    """Parallel (numba prange) shadow-grid driver: ray-traces every (alfa, beta) pixel and fills Mat/Mz.

    The outer loops over alfa and beta use `prange` so numba distributes
    the per-pixel ray tracing (`func`) across multiple CPU cores -- this is
    the key difference from the serial drivers in the sibling files.

    Args:
        rho0, z0: fixed initial emission point.
        M, MD, b: BH mass, disk mass, disk radius.
        alfa, beta: 1-D arrays of emission angles spanning the image grid.
        hder: finite-difference step forwarded to func/geo.

    Returns:
        (Mat, Mz): 2-D arrays of the final rho, z for each (alfa, beta) pixel.
    """
    Mat = np.zeros((len(alfa), len(beta)))
    Mz = np.zeros((len(alfa), len(beta)))

    for i in prange(len(alfa)):
        for j in prange(len(beta)):
            y = np.array([rho0, dr(rho0, z0, M, MD, b, alfa[i], beta[j]), z0, dthe(rho0, z0, M, MD, b, alfa[i])])
            (Mat[i, j], Mz[i, j]) = func(y, 300.0, -0.02, alfa[i], beta[j], M, rho0, z0, MD, b, hder)

    return (Mat, Mz)


# Driver: solve for the initial observer's rho, build the emission-angle
# grid (only the first quadrant, alfaa/betaa halved), ray-trace it in
# parallel via f_paral, and save the resulting Mat/Mz matrices.
weyl_core.load_matrix("Mat_nu_disk0.1")

M = 0.9
MD = 0.1
z0 = 0.0
b = 6.0
hder = 10**-6
func_initial = lambda R: np.sqrt(gpp_i(R, z0, M, MD, b)) - 15.0

R_initial_guess = 10.0
R_solution = fsolve(func_initial, R_initial_guess)

rho0 = float(R_solution)
print(rho0)

alfaa = np.linspace(-np.arctan(10/15), np.arctan(10/15), 160)
betaa = np.linspace(np.arctan(10/15), -np.arctan(10/15), 160)

alfa = np.linspace(alfaa[0], alfaa[int(len(alfaa)/2) - 1], int(len(alfaa)/2))
beta = np.linspace(betaa[0], betaa[int(len(betaa)/2) - 1], int(len(betaa)/2))

start1 = time.time()
start = time.time()
(Mat, Mz) = f_paral(rho0, z0, M, MD, b, alfa, beta, hder)
end = time.time()
print(end - start)



np.savetxt('Mat', Mat)
np.savetxt('Mz', Mz)

end1 = time.time()
print(end1 - start1)
