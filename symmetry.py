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

This is a straight indentation/documentation pass over the original
Jupyter-notebook export. No behaviour was changed; large commented-out
blocks (earlier classification thresholds) are preserved with an
explanatory label.

#Duplicated: `d1`, `d2`, `xi2`, `nuD`, `nu`, `gpp`, and the `_i` observer-frame
variants are copy-pasted verbatim (module-name aside) from generate_matriz.py
/ test_parallel_SHADOW.py. Here they are only used to solve for the initial
observer's rho via fsolve — the actual shadow classification below works
directly off the pre-computed `Mat`/`Mz` matrices, not off these potentials.
"""


import numpy as np
import time
import math
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.optimize import fsolve

from numba import jit
###################################################################################################
# Potentials (only used below to locate the initial observer's rho via fsolve)
@jit(nopython=True)
def d1(rho, z, M):
    """Distance from (rho,z) to the Schwarzschild BH's upper Weyl rod endpoint (0,+M). #Duplicated: see generate_matriz.py."""
    return np.sqrt(rho**2 + (z - M)**2)


@jit(nopython=True)
def d2(rho, z, M):
    """Distance from (rho,z) to the Schwarzschild BH's lower Weyl rod endpoint (0,-M). #Duplicated: see generate_matriz.py."""
    return np.sqrt(rho**2 + (z + M)**2)

###################################################################################################
@jit(nopython=True)
def xi2(R, z, a):
    """Oblate-spheroidal-like coordinate for the Morgan-Morgan disk potential. #Duplicated: see generate_matriz.py."""
    return (np.sqrt((a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)) / (R**2 + z**2)) / np.sqrt(2))


@jit(nopython=True)
def nuD(R, z, M, a):
    """Morgan-Morgan finite thin-disk contribution to the nu metric potential. #Duplicated: see generate_matriz.py."""
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
    """Total nu potential: m=0 BH only (nuSch), m=1 disk only, m=2 BH+disk sum. #Duplicated: see generate_matriz.py."""
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
def gpp(rho, z, M, MD, b):
    """g_phiphi metric component (BH+disk nu potential, m=2) at (rho,z). #Duplicated: see generate_matriz.py."""
    return rho**2 * math.exp(- nu(rho, z, M, MD, b, 2))

#####################################################################

########      TO CALCULATE THE INITIAL OBSERVER'S RHO       #########

def d1_i(rho, z, M):
    """Observer-frame (non-jitted) copy of d1, used to solve for the initial observer's rho via fsolve. #Duplicated: see generate_matriz.py."""
    return np.sqrt(rho**2 + (z - M)**2)


def d2_i(rho, z, M):
    """Observer-frame (non-jitted) copy of d2. #Duplicated: see generate_matriz.py."""
    return np.sqrt(rho**2 + (z + M)**2)



def xi2_i(R, z, a):
    """Observer-frame (non-jitted) copy of xi2. #Duplicated: see generate_matriz.py."""
    return (np.sqrt((a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)) / (R**2 + z**2)) / np.sqrt(2))



def nuD_i(R, z, M, a):
    """Observer-frame (non-jitted) copy of nuD. #Duplicated: see generate_matriz.py."""
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
    """Observer-frame (non-jitted) copy of nu, used to solve for the initial observer's rho via fsolve.

    #Duplicated: see generate_matriz.py.
    #NOTE: the m==1 branch calls the jitted `nuD`, not this file's own `nuD_i` -- kept as in the original.
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
    """Observer-frame (non-jitted) copy of gpp, used to solve for the initial observer's rho via fsolve. #Duplicated: see generate_matriz.py."""
    return rho**2 * math.exp(- nu_i(rho, z, M, MD, b, 2))


#####################################################################
#####################################################################
#####################################################################


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

# alfa = np.linspace(alfaa[0],alfaa[int(len(alfaa)/2)-1],88)
# beta = np.linspace(betaa[0],betaa[int(len(betaa)/2)-1],88)




Mat = np.loadtxt('Mat')
Mz = np.loadtxt('Mz')
# Mphi = np.loadtxt('Mphi')




M2 = np.zeros((len(Mat), len(Mat[0])))
Mat2 = np.zeros((len(Mat), len(Mat[0])))
Mz2 = np.zeros((len(Mat), len(Mat[0])))
Msym = np.zeros((2*len(Mat), 2*len(Mat[0])))


# Legacy: earlier pixel-classification pass, using a raw-radius threshold
# (M + 0.5*(d1+d2)) to detect capture (<= 2.01*M) vs. beyond-disk (> 6.0*M),
# with a separate un-shifted Mz2. Superseded by the two later classification
# attempts below.
"""
for i in range(len(Mz)):
	for j in range(len(Mz[0])):
		if Mz[i,j] > 49.0:
			Mz2[i,j] = Mz[i,j]-50.0
		else:
			Mz2[i,j] = Mz[i,j]

for i in range(len(Mat)):
	for j in range(len(Mat[0])):
		Mat2[i,j] = M+0.5*(d1(Mat[i,j],Mz2[i,j],M)+d2(Mat[i,j],Mz2[i,j],M))
count = 0
for i in range(len(Mat)):
	for j in range(len(Mat[0])):

		if Mat2[i,j] <= 2.01*M:
    			M2[i,j] = 1
    			count +=1

		elif (Mat2[i,j] > 6.0*M	 and Mz[i,j] > 49.0):
			M2[i,j] = 2


		else:
    			M2[i,j] = 0

"""
# Legacy: second attempt at pixel classification, using a fixed small-radius
# capture threshold (Mat2 <= 0.01) and disk threshold (> b). Also superseded.
"""
for i in range(len(Mz)):
	for j in range(len(Mz[0])):
		if Mz[i,j] > 49.0:
			Mz2[i,j] = Mz[i,j]-50.0
		else:
			Mz2[i,j] = Mz[i,j]

for i in range(len(Mat)):
	for j in range(len(Mat[0])):
		Mat2[i,j] = Mat[i,j]

count = 0
for i in range(len(Mat)):
	for j in range(len(Mat[0])):

		if (Mat2[i,j] <= 0.01 and Mz[i,j] < M):
    			M2[i,j] = 1
    			count +=1

		elif (Mat2[i,j] > b and Mz[i,j] > 49.0):
			M2[i,j] = 2

		else:
    			M2[i,j] = 0
print(count)
"""
# Live classification: Mz values above 49.0 are unshifted (photon escaped
# "backwards" past a wraparound offset of 50), Mat values above 50.0 are
# similarly unshifted; a pixel is "captured" (M2=1) if its unshifted radius
# is <= 0.002, or "beyond the disk" (M2=2) if radius > b and z escaped past
# the wraparound; otherwise "neither" (M2=0).
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

# Reconstruct the full image from the ray-traced quarter by mirroring across
# both the alfa and beta axes (z -> -z and left-right symmetry of the
# BH+disk configuration).
for i in range(0, len(M2)):
    for j in range(0, len(M2[0])):
        Msym[i, j] = M2[i, j]

for i in range(0, len(M2)):
    for j in range(0, len(M2[0])):
        Msym[i, -j - 1] = M2[i, j]

for i in range(0, len(M2)):
    for j in range(0, len(M2[0])):
        Msym[-i - 1, j] = M2[i, j]

for i in range(0, len(M2)):
    for j in range(0, len(M2[0])):
        Msym[-i - 1, -j - 1] = M2[i, j]

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

# c_map = [[1,1,1],[0, 0, 0], [0.192, 0.192, 0.192]]
# c_map = [[1,1,1],[0, 0, 0], [1,0,0]]
# c_map = [[0.192,0.192,0.192],[0,0,0],[1,1,0]]
c_map = [[0.192, 0.192, 0.192], [0, 0, 0]]

cm = matplotlib.colors.ListedColormap(c_map)

plt.imshow(Msym, cmap=cm, extent=[-beta[0], -beta[-1], alfa[0], alfa[-1]])

plt.title("$M_{BN}=$" + str(M) + " , $M_{Disk} =$" + str(MD), size=40)
plt.xlabel("$\\beta$", size=30)
plt.ylabel("$\\alpha$", size=30)

plt.tick_params(axis='both', which='major', labelsize=14)
# plt.legend(loc=10, bbox_to_anchor=(0.85, 0.9), ncol=1,fontsize=17)
# plt.xlim(-np.arctan(10/15),np.arctan(10/15))
# plt.ylim(-np.arctan(10/15),np.arctan(10/15))
# plt.show()
plt.savefig("Test")
