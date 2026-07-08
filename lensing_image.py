"""Post-processing script that turns raw ray-tracer output into a gravitational
lensing image (as opposed to a plain captured/escaped shadow silhouette).

It loads three matrices produced by one of the Weyl-coordinate Schwarzschild
ray tracers in this repo (e.g. test_Z_SHADOW.py / test_parallel_SHADOW.py /
test_symmetry_lensing.py): Mat (an intensity/attenuation-like value used here
only to flag captured/background pixels below a small threshold), Mz (the
sign of the photon's final z-coordinate, i.e. which hemisphere of the disk/
sky it landed on), and Mphi (the photon's final azimuthal angle). Tracking
Mphi per pixel is what makes this a lensing *map* rather than a binary
shadow mask: each escaped pixel carries directional information about where
the light ray actually ended up, not just whether it escaped.

The script wraps Mphi into the canonical [0, 2*pi) range (Mphi2), then
classifies every non-background pixel into one of four quadrants formed by
crossing the Mz sign (upper/lower hemisphere) with which half of the wrapped
Mphi2 range ([0, pi] vs (pi, 2*pi]) the angle falls into. Each quadrant is
assigned its own integer code, which is then rendered as a distinct color in
the final image via a 5-entry ListedColormap (background + 4 quadrants).

Near the bottom there is a large disabled block (kept for reference, wrapped
in a triple-quoted string) containing legacy/alternate logic for an Einstein
ring highlight and a black coordinate grid-line overlay; it is never
executed. The final figure is saved to
"Schwarzschild_WeylCoords_lensing_1000x1000".
"""

# ==================== Imports ====================
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import time
from numba import jit
import cmath
import scipy.integrate as sci
from scipy.optimize import fsolve

# ==================== Grid & data setup ====================
alfa = np.linspace(-np.arctan(10/15),np.arctan(10/15),1000)
beta = np.linspace(np.arctan(10/15),-np.arctan(10/15),1000)

# ==================== Load data ====================
# Mat: intensity/attenuation-like value (used below only to flag background pixels)
# Mz: sign of the photon's final z-coordinate (hemisphere)
# Mphi: photon's final azimuthal angle (the "lensing" directional information)
Mat = np.loadtxt("Mat")
Mz = np.loadtxt("Mz")
Mphi = np.loadtxt("Mphi")

M2 = np.zeros((len(Mat),len(Mat[0])))
Mz2 = np.zeros((len(Mz),len(Mz[0])))
Mphi2 = np.zeros((len(Mphi),len(Mphi[0])))

M = 0.9
MD = 0.1

# Grid-overlay tolerances/bounds, only consumed by the disabled block below
dang = 0.01
dz = 0.06
kmax_z   = 18.0
kmax_phi = 2*np.pi/ (10*np.pi/180)

# ==================== Wrap Mphi into [0, 2*pi) ====================
#Duplicated: identical/near-identical Mphi wrap-into-[0,2*pi) loop appears in
#lensing_r_theta.py, test_Z_SHADOW.py, test_parallel_SHADOW.py,
#test2_Z_SHADOW.py and test_symmetry_lensing.py
for i in range(len(Mphi)):
    for j in range(len(Mphi)):
        if Mphi[i,j] > 2*np.pi:
            Mphi2[i,j] = Mphi[i,j]-int(Mphi[i,j]/(2*np.pi))*2*np.pi

        elif (Mphi[i,j] < 0 and Mphi[i,j] > -2*np.pi):
            Mphi2[i,j] = Mphi[i,j] + 2*np.pi

        elif Mphi[i,j] < -2*np.pi:
            Mphi2[i,j] = Mphi[i,j] - int(Mphi[i,j]/(2*np.pi))*2*np.pi + 2*np.pi

        else:
            Mphi2[i,j] = Mphi[i,j]

# ==================== Classification into quadrants ====================
# Background pixels (Mat below threshold) get code 0; otherwise each pixel is
# assigned one of codes 1-4 based on hemisphere (Mz sign) x azimuth half
# (Mphi2 in [0,pi] vs (pi,2*pi]) -- the four "lensing quadrants".
M2 = np.zeros((len(Mat),len(Mat[0])))
for i in range(len(Mat)):
    for j in range(len(Mat[0])):
        if Mat[i,j] < 0.002:
            M2[i,j] = 0
        else:
            if (Mz[i,j] < 0 and (Mphi2[i,j] >= 0 and Mphi2[i,j] <= np.pi)):
                M2[i,j] = 3 #1

            elif ( Mz[i,j] < 0 and (Mphi2[i,j] > np.pi and Mphi2[i,j] <= 2*np.pi)):
                M2[i,j] = 4 #2

            elif (Mz[i,j] >= 0 and (Mphi2[i,j] > 0  and Mphi2[i,j] <= np.pi)):
                M2[i,j] = 2 #3

            elif ( Mz[i,j] >= 0 and (Mphi2[i,j] > np.pi and Mphi2[i,j] <= 2*np.pi )):
                M2[i,j] = 1 #4

# ==================== Grid overlay (disabled/legacy) ====================
###################         FIELD LINES            ###############################
# Legacy/disabled block (never executed, kept for reference): an alternate
# Einstein-ring highlight condition (itself commented out inside) plus a
# black coordinate grid-line overlay that would zero out M2 along constant-z
# and constant-phi lines using the dz/dang tolerances and kmax_z/kmax_phi
# bounds defined above.
"""
for i in range(len(Mat)):
	for j in range(len(Mat[0])):
        	"Einstein ring"
		#if (( ((Mthet2[i,j] < np.pi/2 + 10*np.pi/180 and Mthet2[i,j]> np.pi/2-10*np.pi/180) and \
		#     (Mphi2[i,j]< np.pi + 10*np.pi/180 and Mphi2[i,j]> np.pi -10*np.pi/180)) and \
		#   ((Mphi2[i,j]-np.pi)**2 + (Mthet2[i,j]-np.pi/2)**2 <= np.pi * (np.pi*10/180)**2) ) and \
		#   Mat[i,j] >=0.02):
		#        M2[i,j] = 5

        	"Black lines of the grid"
        	kmax = kmax_z
        	for k in range(0,int(kmax)+1):
            		if (Mz[i,j] <= 30 - k*60.0/18.0 + dz and Mz[i,j] >= 30 - k*60.0/18.0 - dz):
                		M2[i,j] = 0

        	kmax = kmax_phi
        	for k in range(0,int(kmax)+1):
            		if (Mphi2[i,j] <= k*10*np.pi/180 + dang and Mphi2[i,j] >= k*10*np.pi/180-dang):
                		M2[i,j] = 0

"""

# ==================== Plotting ====================
plt.figure(figsize = (25,25))

# Disabled alternate colormap (green/red/blue/yellow ordering instead of the active one below)
#c_map = [[0,0,0],[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0]]#,[1,1,1]]#,[0,1,1]]
c_map = [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0]]

cm = matplotlib.colors.ListedColormap(c_map)

# Disabled alternate imshow calls: one using a gpp(...)-scaled extent (function not
# defined in this file), another rendering a different matrix (M22) in binary
#plt.imshow(M2, cmap = cm,extent=[-np.sqrt(gpp(r0,a,M,theta))*beta[0],-np.sqrt(gpp(r0,a,M,theta))*beta[-1],np.sqrt(gpp(r0,a,M,theta))*alfa[0],np.sqrt(gpp(r0,a,M,theta))*alfa[-1]])
#plt.imshow(M22, cmap = "binary",extent=[beta[0],beta[-1],alfa[0],alfa[-1]])

plt.imshow(M2, cmap = cm,extent=[-beta[0],-beta[-1],alfa[0],alfa[-1]])


plt.title("$M_{BN}=$" + str(M) + " , $M_{Disk} =$" + str(MD) + " , a = 6",size=40)
plt.xlabel("$\\beta$",size=30)
plt.ylabel("$\\alpha$",size=30)

plt.tick_params(axis='both', which='major', labelsize=30)
# Disabled interactive display call (figure is saved to file instead, below)
#plt.show()
plt.savefig("Schwarzschild_WeylCoords_lensing_1000x1000")
