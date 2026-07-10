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

Mirroring and classification are delegated to shadow_postprocess.mirror_quadrants
and shadow_postprocess.classify_shadow (shared helpers also used elsewhere in
this repo) instead of duplicating the loops inline. The alternate/legacy
snippets from the original Jupyter-notebook export (earlier classification
condition, alternate figure size, alternate colormap, disabled axis
labels/tick params, disabled savefig call) have been removed; none of them
were ever executed, so no behaviour was changed.

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

from shadow_postprocess import mirror_quadrants, classify_shadow


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
Mat2 = mirror_quadrants(Mat)
Mz2 = mirror_quadrants(Mz, antisymmetric=True)


###################################################################
# Classification
###################################################################
M2 = classify_shadow(Mat2, Mz2, b)

###################################################################
# Plotting
###################################################################
## FIGURE ##

fig, ax = plt.subplots(1, 1, figsize=(30, 30))


c_map = [[0.505, 0.505, 0.505], [0, 0, 0], [1, 1, 1]]


cm = matplotlib.colors.ListedColormap(c_map)


plt.imshow(M2, cmap=cm, extent=[-beta[0], -beta[-1], alfa[0], alfa[-1]])
ax.add_patch(mpatches.Circle((0.006, 0.001), 0.005, color='blue'))
ax.add_patch(mpatches.Circle((0.006, 0.100), 0.005, color='orange'))
ax.add_patch(mpatches.Circle((0.006, 0.207), 0.005, color='green'))
ax.add_patch(mpatches.Circle((0.006, 0.220), 0.005, color='red'))
ax.add_patch(mpatches.Circle((0.006, 0.300), 0.005, color='purple'))


plt.ylim(0.0, np.arctan(10/15))

plt.title("$m=$" + str(round(MD, 1)), size=84)

plt.show()
