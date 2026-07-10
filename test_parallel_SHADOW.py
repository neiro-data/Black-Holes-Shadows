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

This is a straight indentation/documentation pass over the original
Jupyter-notebook export (note the `# In[NN]` cell markers, kept for
provenance); no behaviour was changed beyond the extraction itself.
"""

# In[25]:


import math
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors
import time
from numba import jit
from numba import prange
import cmath
import scipy.integrate as sci
from scipy.optimize import fsolve

import weyl_core
from weyl_core import *


# In[33]:


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






M2 = np.zeros((len(Mat), len(Mat[0])))
Mat2 = np.zeros((len(alfa), len(beta)))
Mz2 = np.zeros((len(alfa), len(beta)))



# Legacy: earlier pixel-classification pass (raw-radius capture/disk
# thresholds) and its matplotlib figure, superseded by the second attempt
# below and ultimately by the standalone plotting scripts.
"""
for i in range(len(Mz)):
	for j in range(len(Mz[0])):
		if Mz[i,j] > 49.0:
			Mz2[i,j] = Mz[i,j]-50.0
		else:
			Mz2[i,j] = Mz[i,j]

for i in range(len(alfa)):
	for j in range(len(beta)):
		Mat2[i,j] = M+0.5*(d1(Mat[i,j],Mz2[i,j],M)+d2(Mat[i,j],Mz2[i,j],M))
count = 0
for i in range(len(Mat)):
	for j in range(len(Mat[0])):

		if Mat2[i,j] <= 2.01*M:
    			M2[i,j] = 1
    			count +=1

		elif (Mat2[i,j] > 6.0*M and Mz[i,j] > 49.0):
			M2[i,j] = 2

		else:
    			M2[i,j] = 0
print(count)

"""

# Legacy: second pixel-classification + plotting attempt (small-radius
# capture threshold, disk threshold), also superseded.
"""
for i in range(len(Mz)):
	for j in range(len(Mz[0])):
		if Mz[i,j] > 49.0:
			Mz2[i,j] = Mz[i,j]-50.0
		else:
			Mz2[i,j] = Mz[i,j]

for i in range(len(Mat)):
	for j in range(len(Mat[0])):
		if Mat[i,j] > 50.0:
			Mat2[i,j] = Mat[i,j] -50.0
		else:
			Mat2[i,j] = Mat[i,j]
count1 = 0
count2 = 0
for i in range(len(Mat)):
	for j in range(len(Mat[0])):

		if (Mat2[i,j] <= 0.002):# and Mz[i,j] < M):
    			M2[i,j] = 1
    			count1 +=1

		elif (Mat2[i,j] > b and Mz[i,j] > 49.0):
			M2[i,j] = 2


		else:
    			M2[i,j] = 0
print(count1,count2)


plt.figure(figsize = (15,15))


#c_map = [[1,1,1],[0, 0, 0], [0.192, 0.192, 0.192],[1,0,0]]
#c_map = [[1,1,1],[0, 0, 0], [0.192, 0.192, 0.192]]
c_map = [[1,1,1],[0,0,0]]
cm = matplotlib.colors.ListedColormap(c_map)

plt.imshow(M2, cmap = cm,extent=[-beta[0],-beta[-1],alfa[0],alfa[-1]])
#plt.colorbar()


plt.xlabel("$\\beta$",size=30)
plt.ylabel("$\\alpha$",size=30)

plt.tick_params(axis='both', which='major', labelsize=14)
#plt.legend(loc=10, bbox_to_anchor=(0.85, 0.9), ncol=1,fontsize=17)

#plt.show()
plt.savefig("Schwarzschild_Weyl_coords_154x154")


"""

# Legacy: quadrant-based lensing classification + grid-line overlay +
# figure (needs Mphi, never computed by this file's f_paral/func, which
# only track rho/z) -- dead/inert.
"""

Mphi2 = np.zeros((len(alfa),len(beta)))
Mthet2 = np.zeros((len(alfa),len(beta)))
Mat2 = np.zeros((len(alfa),len(beta)))

for i in range(len(alfa)):
	for j in range(len(beta)):
        	Mat2[i,j] = 1+0.5*(d1(Mat[i,j],Mz[i,j],1)+d2(Mat[i,j],Mz[i,j],1))

for i in range(len(alfa)):
	for j in range(len(beta)):
        #Mthet2[i,j] = np.arccos(Mz[i,j]/(np.sqrt(Mat[i,j]**2+Mz[i,j]**2)-1))
        	Mthet2[i,j] = np.arccos(0.5*(np.sqrt(Mat[i,j]**2+(Mz[i,j]+1)**2)-np.sqrt(Mat[i,j]**2+(Mz[i,j]-1)**2)))

for i in range(len(alfa)):
	for j in range(len(beta)):
        	if Mphi[i,j] > 2*np.pi:
            		Mphi2[i,j] = Mphi[i,j]-int(Mphi[i,j]/(2*np.pi))*2*np.pi

        	elif (Mphi[i,j] < 0 and Mphi[i,j] > -2*np.pi):
            		Mphi2[i,j] = Mphi[i,j] + 2*np.pi

        	elif Mphi[i,j] < -2*np.pi:
            		Mphi2[i,j] = Mphi[i,j] - int(Mphi[i,j]/(2*np.pi))*2*np.pi + 2*np.pi

        	else:
            		Mphi2[i,j] = Mphi[i,j]


dang = 0.01
kmax_the = np.pi/ (10*np.pi/180)
kmax_phi = 2*np.pi/ (10*np.pi/180)

M2 = np.zeros((len(Mat),len(Mat[0])))
for i in range(len(Mat)):
	for j in range(len(Mat[0])):
        #if (Mat[i,j] < 0.08 and np.abs(Mz[i,j]) < 1.01) :
        	if Mat2[i,j] < 2.2:
            		M2[i,j] = 0
        	else:
            		if ((Mthet2[i,j] >= np.pi/2 and Mthet2[i,j] <= np.pi) and (Mphi2[i,j] >= 0 and Mphi2[i,j] <= np.pi)):
                		M2[i,j] = 3 #1

            		elif ((Mthet2[i,j] >= np.pi/2  and Mthet2[i,j] <= np.pi) and (Mphi2[i,j] > np.pi and Mphi2[i,j] <= 2*np.pi)):
                		M2[i,j] = 4 #2

            		elif ((Mthet2[i,j] >= 0 and Mthet2[i,j] < np.pi/2) and (Mphi2[i,j] > 0  and Mphi2[i,j] <= np.pi)):
                		M2[i,j] = 2 #3

            		elif ((Mthet2[i,j] >= 0 and Mthet2[i,j] <= np.pi/2) and (Mphi2[i,j] > np.pi and Mphi2[i,j] <= 2*np.pi )):
                		M2[i,j] = 1 #4


for i in range(len(Mat)):
	for j in range(len(Mat[0])):
        	"Einstein ring"
		#if (( ((Mthet2[i,j] < np.pi/2 + 10*np.pi/180 and Mthet2[i,j]> np.pi/2-10*np.pi/180) and \
		#     (Mphi2[i,j]< np.pi + 10*np.pi/180 and Mphi2[i,j]> np.pi -10*np.pi/180)) and \
		#   ((Mphi2[i,j]-np.pi)**2 + (Mthet2[i,j]-np.pi/2)**2 <= np.pi * (np.pi*10/180)**2) ) and \
		#   Mat[i,j] >=0.02):
		#        M2[i,j] = 5

        	"Black lines of the grid"
        	kmax = kmax_the
        	for k in range(0,int(kmax)+1):
            		if (Mthet2[i,j] <= k*10*np.pi/180 + dang and Mthet2[i,j] >= k*10*np.pi/180-dang):
                		M2[i,j] = 0

        	kmax = kmax_phi
        	for k in range(0,int(kmax)+1):
            		if (Mphi2[i,j] <= k*10*np.pi/180 + dang and Mphi2[i,j] >= k*10*np.pi/180-dang):
                		M2[i,j] = 0




plt.figure(figsize = (25,25))

#c_map = [[0,0,0],[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0]]#,[1,1,1]]#,[0,1,1]]
c_map = [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0]]

cm = matplotlib.colors.ListedColormap(c_map)

#plt.imshow(M2, cmap = cm,extent=[-np.sqrt(gpp(r0,a,M,theta))*beta[0],-np.sqrt(gpp(r0,a,M,theta))*beta[-1],np.sqrt(gpp(r0,a,M,theta))*alfa[0],np.sqrt(gpp(r0,a,M,theta))*alfa[-1]])
#plt.imshow(M22, cmap = "binary",extent=[beta[0],beta[-1],alfa[0],alfa[-1]])

plt.imshow(M2, cmap = cm,extent=[-beta[0],-beta[-1],alfa[0],alfa[-1]])

#plt.plot(x(r,a,M,theta,r0),yplus(r,a,M,theta,r0),'o',color="red", label="analytical")
#plt.plot(x(r,a,M,theta,r0),yminus(r,a,M,theta,r0),'o',color="red",label = "analytical")

plt.xlabel("x(M)",size=40)
plt.ylabel("y(M)",size=40)

plt.tick_params(axis='both', which='major', labelsize=30)
#plt.legend(loc=10, bbox_to_anchor=(0.85, 0.9), ncol=1,fontsize=17)
#plt.xlim(-np.arctan(10/15),np.arctan(10/15))
#plt.ylim(-np.arctan(10/15),np.arctan(10/15))
#plt.show()
plt.savefig("Schwarzschild_WeylCoords_lensing_154x154")



"""


# Legacy: single-ray diagnostic run (traces one photon at a fixed alfa/beta
# and plots rho, z, nu, and velocity components along the trajectory to
# sanity-check the integrator), superseded by the grid-driver above.
"""


start = time.time()
M = 0.9
MD = 0.1
rho = 15.0
z0 = 0.0
b = 6.0
rho0 = np.sqrt(gpp(rho,z0,M,MD,b))
hder = 10**-6


#alfa = -0.3311738801589748
#beta = 0.006758650615489303
alfa = -0.2
beta = 0.006758650615489303

y = np.array([rho0,dr(rho0,z0,M,MD,b,alfa,beta), z0,dthe(rho0,z0,M,MD,b,alfa)])

X = (func(y,300.0,-0.02,alfa,beta,M,rho0,z0,MD,b,hder)[0])
Y1 = (func(y,300.0,-0.02,alfa,beta,M,rho0,z0,MD,b,hder)[1])
Y2 = (func(y,300.0,-0.02,alfa,beta,M,rho0,z0,MD,b,hder)[2])
Y3 = (func(y,300.0,-0.02,alfa,beta,M,rho0,z0,MD,b,hder)[3])
Y4 = (func(y,300.0,-0.02,alfa,beta,M,rho0,z0,MD,b,hder)[4])

end = time.time()
print(end-start)


# In[24]:

print("rho_min=",np.min(Y1))
print("r_min=",np.min( M+0.5*(d1(np.transpose(Y1),np.transpose(Y3),M)+d2(np.transpose(Y1),np.transpose(Y3),M)) ))
plt.figure(figsize=(20,20))
NU = np.zeros(len(X))
DRHO = np.zeros(len(X))
DZ = np.zeros(len(X))
dNU = np.zeros(len(X))
for i in range(len(X)):
#    dNU[i] = derNU(np.transpose(Y1)[i],np.transpose(Y3)[i],1.0,1.0,6.0,1,2,10**-6)
	NU[i] = nu(np.transpose(Y1)[i],np.transpose(Y3)[i],M,MD,b,0)
	DRHO[i] = dr(np.transpose(Y1)[i],np.transpose(Y3)[i],M,MD,b,alfa,beta)
	DZ[i] = dthe(np.transpose(Y1)[i],np.transpose(Y3)[i],M,MD,b,alfa)
print(np.min(NU))
plt.plot(np.transpose(X),NU,'.',label="nu")
plt.plot(np.transpose(X),DRHO,'.',label="DRHO")
plt.plot(np.transpose(X),DZ,'.', label="DZ")
#plt.plot(np.transpose(X),dNU,'.',label="dNU")
#plt.plot(np.transpose(X),derNU(np.transpose(Y3),np.transpose(Y1),1,1,1,1,1,10**-8),'.',label="dNUz")
#plt.plot(np.transpose(X),dlambr(np.transpose(Y1),np.transpose(Y3),1,1,1,1,1,10**-8),'.',label="dlambr")
#plt.plot(np.transpose(X),dlambz(np.transpose(Y3),np.transpose(Y1),1,1,1,1,1,10**-8),'.',label="dlambz")
#plt.plot(np.transpose(X),nu(np.transpose(Y1),np.transpose(Y3),1,1,1,1,1),'.',label="NU")
#plt.plot(np.transpose(X),lamb(np.transpose(Y1),np.transpose(Y3),1,1,1,1,1),'.',label="LAMB")



plt.plot(np.transpose(X),np.transpose(Y1),'.',label= "rho")
#plt.plot(np.transpose(X),np.transpose(Y2),'.',label="drho")
plt.plot(np.transpose(X),np.transpose(Y3),'.',label = "z")
#plt.plot(np.transpose(X),np.transpose(Y4),'.',label="dz")

#plt.axhline(M*2.0)
plt.axhline(0.0)
plt.plot(np.transpose(X),M+0.5*(d1(np.transpose(Y1),np.transpose(Y3),M)+d2(np.transpose(Y1),np.transpose(Y3),M)),'.',label="r")

plt.legend(loc=10, bbox_to_anchor=(0.9, 0.8), ncol=2,fontsize=20)
plt.xlabel("lambda = proper time",size = 16)
plt.ylabel("$\\rho$", size = 16)
plt.ylim(-30,30)
#plt.axhline(0.0, color = "red")
#plt.axhline(1.0, color = "green")
#plt.axhline(2.0)
#plt.xlim(280,300)
#plt.ylim(-2,3)
plt.show()


"""
