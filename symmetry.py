#!/usr/bin/env python
# coding: utf-8
"""Build a 4-fold-symmetric black-hole shadow image from a ray-traced quarter.

Post-processing/plotting script for the Weyl-coordinate Schwarzschild BH +
Morgan-Morgan disk family (see generate_matriz.py, test_Z_SHADOW.py and
siblings for the ray tracers). Loads the `Mat`/`Mz` matrices produced by one
of those ray tracers (np.loadtxt), classifies each pixel as "captured by the
BH" / "beyond the disk" / "neither", then reconstructs the full image by
mirroring the traced quarter across both axes (the physical setup is
symmetric under z -> -z and alfa/beta sign flips, so only one quadrant needs
to be ray-traced). Saves the resulting figure ("Test").

Physics helpers (`d1`, `d2`, `xi2`, `nuD`, `nu`, `gpp` and their `_i`
observer-frame variants) and the pixel-classification/mirroring loops now
live in the shared `general_methods` package and `shadow_postprocess.py`
respectively, rather than being copy-pasted locally.
"""


import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.optimize import fsolve

import general_methods
from general_methods import *
from shadow_postprocess import mirror_quadrants, classify_shadow

##############################################################################################################

# Driver: load a quarter-image of the shadow (Mat, Mz) produced by one of the
# ray tracers, classify each pixel into "captured"/"beyond disk"/"neither",
# then reconstruct the full symmetric image by mirroring across both axes.
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

alfa = np.linspace(-np.arctan(10/15), np.arctan(10/15), 752)
beta = np.linspace(np.arctan(10/15), -np.arctan(10/15), 752)

Mat = np.loadtxt('Mat')
Mz = np.loadtxt('Mz')




# Live classification: Mz values above 49.0 are unshifted (photon escaped
# "backwards" past a wraparound offset of 50), Mat values above 50.0 are
# similarly unshifted; a pixel is "captured" (M2=1) if its unshifted radius
# is <= 0.002, or "beyond the disk" (M2=2) if radius > b and z escaped past
# the wraparound; otherwise "neither" (M2=0).
M2 = classify_shadow(Mat, Mz, b, unshift_mat=True, use_abs_z=False)

# Reconstruct the full image from the ray-traced quarter by mirroring across
# both the alfa and beta axes (z -> -z and left-right symmetry of the
# BH+disk configuration).
Msym = mirror_quadrants(M2)

count = 0
for i in range(len(Msym)):
    for j in range(len(Msym)):
        if Msym[i, j] == 1:
            count += 1
        else:
            count += 0

print(count)

end = time.time()
print(end - start)
plt.figure(figsize=(15, 15))

c_map = [[0.192, 0.192, 0.192], [0, 0, 0]]

cm = matplotlib.colors.ListedColormap(c_map)

plt.imshow(Msym, cmap=cm, extent=[-beta[0], -beta[-1], alfa[0], alfa[-1]])

plt.title("$M_{BN}=$" + str(M) + " , $M_{Disk} =$" + str(MD), size=40)
plt.xlabel("$\\beta$", size=30)
plt.ylabel("$\\alpha$", size=30)

plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig("Test")
