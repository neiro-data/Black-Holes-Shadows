#!/usr/bin/env python
"""Load a saved Mat/Mz/Mphi triple from test_symmetry_lensing.py and plot per-quadrant colours."""

import matplotlib.pyplot as plt
import numpy as np

from weyl_plot import classify_lensing_quadrants, get_colormap

if __name__ == "__main__":
    alfa = np.linspace(-np.arctan(10 / 15), np.arctan(10 / 15), 1000)
    beta = np.linspace(np.arctan(10 / 15), -np.arctan(10 / 15), 1000)

    Mat = np.loadtxt("Mat")
    Mz = np.loadtxt("Mz")
    Mphi = np.loadtxt("Mphi")

    M = 0.9
    MD = 0.1

    M2 = classify_lensing_quadrants(Mat, Mz, Mphi)

    plt.figure(figsize=(25, 25))
    cm = get_colormap("quadrant")
    plt.imshow(M2, cmap=cm, extent=[-beta[0], -beta[-1], alfa[0], alfa[-1]])

    plt.title("$M_{BN}=$" + str(M) + " , $M_{Disk} =$" + str(MD) + " , a = 6", size=40)
    plt.xlabel("$\\beta$", size=30)
    plt.ylabel("$\\alpha$", size=30)
    plt.tick_params(axis="both", which="major", labelsize=30)
    plt.savefig("Schwarzschild_WeylCoords_lensing_1000x1000")
