#!/usr/bin/env python
"""Variant of simetria_shadow.py: crop to the lower half and save (instead of show)."""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from weyl_plot import get_colormap, mirror_raw_and_classify

if __name__ == "__main__":
    Mat = np.loadtxt("Mat")
    Mz = np.loadtxt("Mz")

    M = 0.4
    MD = 0.6
    b = 3.0

    alfa = np.linspace(-np.arctan(10 / 15), np.arctan(10 / 15), 1000)
    beta = np.linspace(np.arctan(10 / 15), -np.arctan(10 / 15), 1000)

    M2 = mirror_raw_and_classify(Mat, Mz, b)

    M3 = M2[500 : 500 + int(len(M2) / 2), :]

    fig, ax = plt.subplots(1, 1, figsize=(30, 30))
    cm = get_colormap("grey_black_white")

    plt.imshow(M3, cmap=cm, extent=[-beta[0], -beta[-1], alfa[0], 0.0], aspect="equal")
    ax.add_patch(mpatches.Circle((0.006, -0.005), 0.007, color="blue"))
    plt.plot(0.006, -0.100, "v", color="orange", ms=20)
    plt.plot(0.006, -0.207, "s", color="green", ms=20)
    plt.plot(0.006, -0.220, "+", color="red", mew=5, ms=20)
    plt.plot(0.006, -0.300, "x", color="purple", mew=5, ms=20)

    plt.axis("off")
    plt.savefig("M=0_4")
