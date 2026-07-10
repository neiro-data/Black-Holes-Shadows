"""Post-processing/plotting script for the Weyl-coordinate Schwarzschild BH +
Morgan-Morgan disk shadow simulation.

Loads the precomputed Mat (impact-parameter/deflection magnitude) and Mz
(z-related quantity) matrices produced by one of the ray tracers (e.g.
test_Z_SHADOW.py / test2_Z_SHADOW.py / test_parallel_SHADOW.py /
generate_matriz.py) via np.loadtxt, mirrors each quarter-plane result into a
full-plane image (Mat2/Mz2, each twice the size of Mat/Mz) using index
mirroring, classifies every pixel of the mirrored image into M2 (0 = neither,
1 = captured by the black hole, 2 = beyond the disk), overlays marker
annotations at fixed points of interest, and saves the resulting matplotlib
figure to disk.

This is a "v2" variant of simetria_shadow.py. Concrete differences observed
in this version versus v1:
  - After building M2, this script additionally crops it down to M3, keeping
    only the bottom half of the rows (i.e. only one of the two mirrored
    alfa-halves is plotted), whereas v1 plots the full M2 directly.
  - The imshow extent/aspect and axis handling differ: v2 uses M3 with
    aspect='equal' and turns the axes off entirely (plt.axis('off')) instead
    of setting explicit ylim/title/labels like v1.
  - v2 replaces most of v1's circle-patch markers (mpatches.Circle) with
    plt.plot marker calls using distinct marker shapes/colors (triangle,
    square, plus, cross) at the same points of interest; only the first blue
    circle patch is kept as-is. The original circle-patch calls for the other
    points are preserved below as a commented-out (triple-quoted) block
    rather than removed.
  - v2 saves the figure to a file ("M=0_4") via plt.savefig instead of
    calling plt.show() as v1 does.
"""

# ===== Imports =====
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors

import matplotlib.patches as mpatches
from matplotlib.figure import figaspect

from shadow_postprocess import mirror_quadrants, classify_shadow

# ===== Load data =====
Mat = np.loadtxt("Mat")
Mz = np.loadtxt("Mz")

M = 0.4
MD = 0.6
b = 3.0

alfa = np.linspace(-np.arctan(10/15),np.arctan(10/15),1000)
beta = np.linspace(np.arctan(10/15),-np.arctan(10/15),1000)


# ===== Mirror to full plane =====
Mat2 = mirror_quadrants(Mat)
Mz2 = mirror_quadrants(Mz, antisymmetric=True)


# ===== Classification =====
M2 = classify_shadow(Mat2, Mz2, b)


# Crop the mirrored classification grid down to its bottom half (rows
# 500..999), keeping only one of the two mirrored alfa-halves for plotting.
M3 = np.zeros((int(len(M2)/2),len(M2[0])))

for i in range(int(len(M2)/2)):
    for j in range(len(M2[0])):
        M3[i,j] =  M2[i+500,j]
## FIGURE ##

# ===== Plotting + annotations =====
fig,ax = plt.subplots(1,1,figsize=(30,30))


c_map = [[0.505, 0.505, 0.505],[0, 0, 0], [1,1,1]]


cm = matplotlib.colors.ListedColormap(c_map)


plt.imshow(M3, cmap = cm,extent=[-beta[0],-beta[-1],alfa[0],0.0],aspect='equal')
ax.add_patch(mpatches.Circle((0.006,-0.005),0.007,color='blue'))
plt.plot(0.006, -0.100, 'v', color='orange', ms=20)
plt.plot(0.006, -0.207, 's', color='green', ms=20)
plt.plot(0.006, -0.220, '+', color='red', mew=5, ms=20)
plt.plot(0.006, -0.300, 'x', color='purple',mew=5, ms=20)



plt.axis('off')

plt.savefig("M=0_4")
