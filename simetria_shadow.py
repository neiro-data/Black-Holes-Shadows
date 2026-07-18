#!/usr/bin/env python
"""Load a saved Mat/Mz pair, mirror to 4 quadrants, classify, and plot with marker overlays."""

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

    fig, ax = plt.subplots(1, 1, figsize=(30, 30))
    cm = get_colormap("grey_black_white")

    plt.imshow(M2, cmap=cm, extent=[-beta[0], -beta[-1], alfa[0], alfa[-1]])
    ax.add_patch(mpatches.Circle((0.006, 0.001), 0.005, color="blue"))
    ax.add_patch(mpatches.Circle((0.006, 0.100), 0.005, color="orange"))
    ax.add_patch(mpatches.Circle((0.006, 0.207), 0.005, color="green"))
    ax.add_patch(mpatches.Circle((0.006, 0.220), 0.005, color="red"))
    ax.add_patch(mpatches.Circle((0.006, 0.300), 0.005, color="purple"))

    plt.ylim(0.0, np.arctan(10 / 15))
    plt.title("$m=$" + str(round(MD, 1)), size=84)
    plt.show()
