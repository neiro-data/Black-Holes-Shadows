#!/usr/bin/env python
# coding: utf-8
"""Symmetry-exploiting gravitational-lensing ray tracer for the Schwarzschild
BH + Morgan-Morgan disk system, in Weyl coordinates.

Unlike the plain shadow tracers in this repo (test_Z_SHADOW.py,
test_parallel_SHADOW.py), this file builds a gravitational-lensing *image*
rather than just a binary capture/escape silhouette: its geodesic state
vector carries an extra phi component, and its single-ray tracer `func`
returns the photon's final (rho, z, phi), not just (rho, z). That final phi
is what lets downstream plotting distinguish which patch of sky/disk a given
pixel's light ray actually originated from.

To cut compute, this file exploits the up-down and left-right reflection
symmetry of the Weyl-coordinate geometry: the driver at the bottom only
ray-traces one quadrant of the (alfa, beta) emission-angle grid (each of
`alfaa`/`betaa` is halved before tracing). A large, inert, triple-quoted
commented-out block further down contains the intended mirroring/
reconstruction step -- it rebuilds the full four-quadrant matrices from the
traced quadrant via index reflections (with a sign flip for z and a
`2*pi - phi` wrap for the angular coordinate), plus legacy pixel
classification and matplotlib plotting code. None of that block executes.

The shared physics core (simps, derivative, d1, d2, xi2, nuD, nu, lambSch,
gpp/gtt/grr/gzz, zeta, dthe, dr, Pphi, Pt, dphi, dt, the `_i` observer-frame
variants, derNU, dlamb, dlamb2, lamb, run_kut4_mod) now lives in weyl_core.py
and is imported below -- see weyl_core.py's module docstring for the
canonicalization notes (xi2's np.abs guard, unified decorators). `geo` and
`func` remain local to this file: they carry the extra phi state component
that makes this a lensing-map builder rather than a binary shadow tracer --
this file's genuine variant behaviour, not shared with the other tracers.

This file depends on a pre-tabulated lambda-potential matrix loaded via
`weyl_core.load_matrix("Mat_nu_disk0.1")` (produced by generate_matriz.py),
and saves its output matrices (`Mat`, `Mz`, `Mphi`) via np.savetxt for the
plotting scripts (symmetry.py, simetria_shadow*.py, lensing_image.py) to
consume.

This is a straight indentation/documentation pass over the original
Jupyter-notebook export (note the `# In[NN]` cell markers, kept for
provenance); no behaviour was changed beyond the extraction itself.
"""

# In[25]:


import math
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.colors
import time
from numba import jit
import cmath
import scipy.integrate as sci
from scipy.optimize import fsolve

import weyl_core
from weyl_core import *


# In[33]:


# Geodesic & ray tracing
@jit(nopython=True)
def geo(t,z,M,alfa,beta,rho0,z0,MD,b,hder):
    """Geodesic equations of motion (right-hand side), via the Weyl nu/lambda potentials and their derivatives.

    Args:
        t: current affine parameter (unused directly, kept for the RKF45 interface).
        z: state vector [rho, drho/dtau, z_coord, dz/dtau, phi].
        M, MD, b: BH mass, disk mass, disk radius.
        alfa, beta: photon emission angles at the initial point, used (with
            rho0, z0) to compute the conserved momenta pphi, pt.
        rho0, z0: initial emission point (kept fixed since pphi, pt are
            constants of motion).
        hder: finite-difference step used inside derNU.

    Returns:
        np.array([drho, d2rho, dz, d2z, dphi]) -- the derivative of the
        state vector z, i.e. the 2nd-order radial/z photon equations plus
        the phi "velocity", derived from the Weyl metric potentials.
    """

    pphi = Pphi(rho0,z0,M,MD,b,alfa,beta)
    pt = Pt(rho0,z0,M,MD,b,alfa,beta)

    d2Rdt = -(0.5*math.exp(2*nu(z[0],z[2],M,MD,b,2)-lamb(z[0],z[2],M,MD,b,2))*derNU(z[0],z[2],M,MD,b,0,2,hder)* dt(z[0],z[2],M,MD,b,alfa,beta,pt)**2              +0.5*(dlamb2(z[0],z[2],M,MD,b,0,2,hder)-derNU(z[0],z[2],M,MD,b,0,2,hder))*z[1]**2 +               (dlamb2(z[0],z[2],M,MD,b,1,2,hder)-derNU(z[0],z[2],M,MD,b,1,2,hder))*z[1]*z[3]              -0.5*(dlamb2(z[0],z[2],M,MD,b,0,2,hder)-derNU(z[0],z[2],M,MD,b,0,2,hder))*z[3]**2              +0.5*math.exp(-lamb(z[0],z[2],M,MD,b,2))*z[0]*(-2+z[0]*derNU(z[0],z[2],M,MD,b,0,2,hder))*dphi(z[0],z[2],M,MD,b,alfa,beta,pphi)**2)

    d2Thedt = -(0.5*math.exp(2*nu(z[0],z[2],M,MD,b,2)-lamb(z[0],z[2],M,MD,b,2))*derNU(z[0],z[2],M,MD,b,1,2,hder)* dt(z[0],z[2],M,MD,b,alfa,beta,pt)**2                -0.5*(dlamb2(z[0],z[2],M,MD,b,1,2,hder)-derNU(z[0],z[2],M,MD,b,1,2,hder))*z[1]**2                + (dlamb2(z[0],z[2],M,MD,b,0,2,hder)-derNU(z[0],z[2],M,MD,b,0,2,hder))*z[1]*z[3]                + 0.5*(dlamb2(z[0],z[2],M,MD,b,1,2,hder)-derNU(z[0],z[2],M,MD,b,1,2,hder))*z[3]**2                + 0.5*math.exp(-lamb(z[0],z[2],M,MD,b,2))*z[0]**2*derNU(z[0],z[2],M,MD,b,1,2,hder)*dphi(z[0],z[2],M,MD,b,alfa,beta,pphi)**2)

    dPhidt = 1/gpp(z[0], z[2],M,MD,b)*pphi



    return np.array([z[1], d2Rdt, z[3], d2Thedt,dPhidt])





@jit(nopython=True)
def func(y,x,h,alfa,beta,M,rho0,z0,MD,b,hder):
    """Single-ray tracer: integrates one photon's geodesic until it escapes, is captured, or exits otherwise.

    Repeatedly advances the state with `run_kut4_mod` while the photon's nu
    potential stays above -3.0 and its areal radius stays below 30 (i.e.
    while it is neither captured nor escaped).

    Args:
        y: initial state [rho0, drho, z0, dz, phi].
        x: initial affine parameter.
        h: initial (negative) step size.
        alfa, beta: emission angles.
        M, rho0, z0, MD, b, hder: BH mass, initial emission point, disk
            mass, disk radius, finite-difference step.

    Returns:
        [rho, z, phi] at the escape point (areal radius >= 30), or the
        sentinel [0.001, 0.001, 0.2] if the photon is captured (nu <= -3.0).
    """

    it = 0
    Y=[]
    Y.append(y)
    while (nu(y[0],y[2],M,MD,b,2) > -3.0 and np.sqrt(gpp(np.sqrt(y[0]**2+y[2]**2),0,M,MD,b)) < 30.0):
        (h,x,y) = run_kut4_mod(geo, x, y, h,M,alfa,beta,rho0,z0,MD,b,hder)
        Y.append(y)
        it += 1


        if np.sqrt(gpp(np.sqrt(y[0]**2+y[2]**2),0,M,MD,b)) >= 30.0:
            yf = [Y[-1][0],Y[-1][2],Y[-1][4]]
            break

        ####   Pontos que caem no buraco negro    #####

        elif nu(y[0],y[2],M,MD,b,2) <= -3.0:
            yf = [0.001,0.001,0.2]
            break



    return (yf)




# Driver: solve for the initial observer's rho, build the emission-angle
# grid but only over a quarter of the full (alfa, beta) range (symmetry
# exploitation: the full lensing image can be reconstructed from one
# quadrant by reflection, done in the dead mirroring block further below),
# ray-trace that quadrant with `func`, and save the resulting Mat/Mz/Mphi
# matrices.
weyl_core.load_matrix("Mat_nu_disk0.1")

start = time.time()
M = 0.9
MD = 0.1
z0 = 0.0
b = 6.0
hder = 10**-6
func_initial = lambda R : np.sqrt(gpp_i(R,z0,M,MD,b)) - 15.0

R_initial_guess = 10.0
R_solution = fsolve(func_initial, R_initial_guess)

rho0 = float(R_solution)
print(rho0)



hder = 10**-6

alfaa = np.linspace(-np.arctan(10/15),np.arctan(10/15),1000)
betaa = np.linspace(np.arctan(10/15),-np.arctan(10/15),1000)

alfa = np.linspace(alfaa[0],alfaa[int(len(alfaa)/2)-1],int(len(alfaa)/2))
beta = np.linspace(betaa[0],betaa[int(len(betaa)/2)-1],int(len(betaa)/2))



Mat = np.zeros((len(alfa),len(beta)))
Mz = np.zeros((len(alfa),len(beta)))
Mphi = np.zeros((len(alfa),len(beta)))

for i in range(len(alfa)):
    for j in range(len(beta)):


        y = np.array([rho0,dr(rho0,z0,M,MD,b,alfa[i],beta[j]), z0,dthe(rho0,z0,M,MD,b,alfa[i]),0])
        (Mat[i,j],Mz[i,j],Mphi[i,j]) = func(y,300.0,-0.02,alfa[i],beta[j],M,rho0,z0,MD,b,hder)


end = time.time()
print(end-start)


np.savetxt('Mat',Mat)
np.savetxt('Mz',Mz)
np.savetxt('Mphi',Mphi)



#############################################################
#############################################################

########            UNCOMMENT           ####################

#############################################################
#############################################################
#############################################################

# Legacy plotting: mirroring/reconstruction block (dead, never executed).
# This is the symmetry step referenced above -- it rebuilds full four-
# quadrant Mat2/Mz2/Mphi3 matrices from the traced quadrant via reflected
# indexing ([i,j], [i,-j-1], [-i-1,j], [-i-1,-j-1]), with a sign flip for z
# and a 2*pi-phi wrap for the angular coordinate, then classifies pixels
# into four lensing-image quadrants (M2) and plots them.
"""
Mat2 = np.zeros((2*len(Mat),2*len(Mat[0])))
Mz2 = np.zeros((2*len(Mz),2*len(Mz[0])))
Mphi2 = np.zeros((len(Mphi),len(Mphi[0])))
Mphi3 = np.zeros((2*len(Mphi2),2*len(Mphi2[0])))
M2 = np.zeros((len(Mat2),len(Mat2[0])))


for i in range(len(Mphi)):
    for j in range(len(Mphi[0])):
            if Mphi[i,j] > 2*np.pi:
                    Mphi2[i,j] = Mphi[i,j]-int(Mphi[i,j]/(2*np.pi))*2*np.pi

            elif (Mphi[i,j] < 0 and Mphi[i,j] > -2*np.pi):
                    Mphi2[i,j] = Mphi[i,j] + 2*np.pi

            elif Mphi[i,j] < -2*np.pi:
                    Mphi2[i,j] = Mphi[i,j] - int(Mphi[i,j]/(2*np.pi))*2*np.pi + 2*np.pi

            else:
                    Mphi2[i,j] = Mphi[i,j]


########    RADIAL COORDINATE   ############

for i in range(len(Mat)):
    for j in range(len(Mat[0])):
        Mat2[i,j] = Mat[i,j]
        Mat2[i,-j-1] = Mat[i,j]
        Mat2[-i-1,j] = Mat[i,j]
        Mat2[-i-1,-j-1] = Mat[i,j]


########    PHI COORDINATE   ############

for i in range(len(Mphi2)):
    for j in range(len(Mphi2[0])):
        Mphi3[i,j]       = Mphi2[i,j]
        Mphi3[i,-j-1]    = 2*np.pi - Mphi2[i,j]
        Mphi3[-i-1,j]    = Mphi2[i,j]
        Mphi3[-i-1,-j-1] = 2*np.pi - Mphi2[i,j]

########    Z COORDINATE   ############

for i in range(len(Mz)):
    for j in range(len(Mz[0])):
        Mz2[i,j] = Mz[i,j]
        Mz2[i,-j-1] = Mz[i,j]
        Mz2[-i-1,j] = - Mz[i,j]
        Mz2[-i-1,-j-1] = -Mz[i,j]


for i in range(len(Mat2)):
    for j in range(len(Mat2[0])):
            if Mat2[i,j] < 0.002:
                    M2[i,j] = 0
            else:
                    if (Mz2[i,j] < 0 and (Mphi3[i,j] >= 0 and Mphi3[i,j] <= np.pi)):
                        M2[i,j] = 3 #1

                    elif ( Mz2[i,j] < 0 and (Mphi3[i,j] > np.pi and Mphi3[i,j] <= 2*np.pi)):
                        M2[i,j] = 4 #2

                    elif (Mz2[i,j] >= 0 and (Mphi3[i,j] > 0  and Mphi3[i,j] <= np.pi)):
                        M2[i,j] = 2 #3

                    elif ( Mz2[i,j] >= 0 and (Mphi3[i,j] > np.pi and Mphi3[i,j] <= 2*np.pi )):
                        M2[i,j] = 1 #4




plt.figure(figsize = (25,25))

#c_map = [[0,0,0],[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0]]#,[1,1,1]]#,[0,1,1]]
c_map = [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0]]

cm = matplotlib.colors.ListedColormap(c_map)

#plt.imshow(M2, cmap = cm,extent=[-np.sqrt(gpp(r0,a,M,theta))*beta[0],-np.sqrt(gpp(r0,a,M,theta))*beta[-1],np.sqrt(gpp(r0,a,M,theta))*alfa[0],np.sqrt(gpp(r0,a,M,theta))*alfa[-1]])
#plt.imshow(M22, cmap = "binary",extent=[beta[0],beta[-1],alfa[0],alfa[-1]])

plt.imshow(M2, cmap = cm,extent=[-beta[0],-beta[-1],alfa[0],alfa[-1]])



plt.xlabel("x(M)",size=40)
plt.ylabel("y(M)",size=40)

plt.tick_params(axis='both', which='major', labelsize=30)

#plt.show()
plt.savefig("Schwarzschild_WeylCoords_lensing_154x154")

#############################################################
#############################################################
#############################################################
#############################################################
#############################################################


"""

# Legacy plotting: second (R, theta)-based lensing classification + grid-
# line overlay + figure attempt, also dead/never executed.
#####################		( R , THETA ) 		###################################
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
