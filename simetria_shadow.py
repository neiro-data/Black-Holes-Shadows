"""Build a full black-hole shadow image from a symmetric quarter and plot it.

Post-processing/plotting script for the Weyl-coordinate Schwarzschild BH +
Morgan-Morgan disk shadow-simulation family in this repo (see generate_matriz.py,
test_Z_SHADOW.py, test2_Z_SHADOW.py, test_parallel_SHADOW.py and
test_symmetry_lensing.py for the ray tracers). Loads the `Mat`/`Mz` matrices
produced by one of those ray tracers (via np.loadtxt), mirrors the traced
quarter across both axes to reconstruct the full symmetric image (Mat2/Mz2,
with Mz2 flipping sign in the mirrored-z quadrants), classifies each pixel of
that full image into "captured by the BH" (M2=1), "beyond the disk" (M2=2) or
"neither" (M2=0) using simple radius/z thresholds, then builds a matplotlib
figure (colormap + imshow, with a handful of decorative colored circles) and
displays it with plt.show().

This is a straight indentation/documentation pass over the original
Jupyter-notebook export. No behaviour was changed; the alternate/legacy
snippets (earlier classification condition, alternate figure size, alternate
colormap, disabled axis labels/tick params, disabled savefig call) are kept
in place with an explanatory comment label.

No `def` functions and no copy-pasted potential-helper code (d1/d2/xi2/nu/gpp)
from symmetry.py / generate_matriz.py are present in this file, so there is
nothing here to tag with a #Duplicated comment.
"""

###################################################################
# Imports
###################################################################
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors

import matplotlib.patches as mpatches


###################################################################
# Load data
###################################################################
Mat = np.loadtxt("Mat")
Mz = np.loadtxt("Mz")

###################################################################
# Constants & angle axes
###################################################################
M = 0.4
MD = 0.6
b = 3.0

alfa = np.linspace(-np.arctan(10/15), np.arctan(10/15), 1000)
beta = np.linspace(np.arctan(10/15), -np.arctan(10/15), 1000)


###################################################################
# Symmetry reconstruction
###################################################################
# Mirror the traced quarter into all four quadrants: Mat2 (radius-like
# quantity) is mirrored unchanged, Mz2 flips sign in the z-mirrored
# (bottom) quadrants since z -> -z is antisymmetric.
Mat2 = np.zeros((2*len(Mat), 2*len(Mat[0])))
Mz2 = np.zeros((2*len(Mz), 2*len(Mz[0])))


for i in range(len(Mat)):
    for j in range(len(Mat[0])):
        Mat2[i, j] = Mat[i, j]
        Mat2[i, -j-1] = Mat[i, j]
        Mat2[-i-1, j] = Mat[i, j]
        Mat2[-i-1, -j-1] = Mat[i, j]

        ##Mz
        Mz2[i, j] = Mz[i, j]
        Mz2[i, -j-1] = Mz[i, j]
        Mz2[-i-1, j] = - Mz[i, j]
        Mz2[-i-1, -j-1] = - Mz[i, j]


###################################################################
# Classification
###################################################################
M2 = np.zeros((len(Mat2), len(Mat2[0])))

for i in range(len(Mat2)):
    for j in range(len(Mat2[0])):

        # Captured by the BH if within a small radius of the origin.
        # The trailing "and Mz[i,j] < M" alternate condition below is an
        # earlier, disabled version of this capture threshold.
        if (Mat2[i, j] <= 0.002):  # and Mz[i,j] < M):
            M2[i, j] = 1

        elif (Mat2[i, j] > b and (Mz2[i, j] > 49.0 or Mz2[i, j] < -49.0)):
            M2[i, j] = 2

        else:
            M2[i, j] = 0

###################################################################
# Plotting
###################################################################
## FIGURE ##

# Alternate figure size tried before switching to fig/ax + add_patch below.
#plt.figure(figsize = (22,22))
fig, ax = plt.subplots(1, 1, figsize=(30, 30))


# Alternate colormap (white/black/grey) tried before the current
# grey/black/white one.
#c_map = [[1,1,1],[0, 0, 0], [0.192, 0.192, 0.192]]
c_map = [[0.505, 0.505, 0.505], [0, 0, 0], [1, 1, 1]]


cm = matplotlib.colors.ListedColormap(c_map)


plt.imshow(M2, cmap=cm, extent=[-beta[0], -beta[-1], alfa[0], alfa[-1]])
ax.add_patch(mpatches.Circle((0.006, 0.001), 0.005, color='blue'))
ax.add_patch(mpatches.Circle((0.006, 0.100), 0.005, color='orange'))
ax.add_patch(mpatches.Circle((0.006, 0.207), 0.005, color='green'))
ax.add_patch(mpatches.Circle((0.006, 0.220), 0.005, color='red'))
ax.add_patch(mpatches.Circle((0.006, 0.300), 0.005, color='purple'))


# Disabled axis labels (kept for reference).
#plt.xlabel("$\\beta$",size=84)
#plt.ylabel("$\\alpha$",size=84)

plt.ylim(0.0, np.arctan(10/15))
# Disabled tick-size tweak.
#plt.tick_params(axis='both', which='major', labelsize=60)

plt.title("$m=$" + str(round(MD, 1)), size=84)

plt.show()
# Alternate save-to-file call, currently disabled in favor of plt.show().
#plt.savefig("M=0_4")
