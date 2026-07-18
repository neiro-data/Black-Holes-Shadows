#!/usr/bin/env python
"""Load a saved Mat/Mz pair, classify shadow/disk, mirror to 4 quadrants, and plot.

Uses the observer-frame helpers from weyl_core (gpp_i etc.) purely to report rho0 for
context; the actual classification only needs the already-traced Mat/Mz matrices.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from weyl_core import solve_observer_rho
from weyl_plot import classify_shadow_disk, get_colormap, mirror_quadrants

if __name__ == "__main__":
    start = time.time()
    M = 1.0
    MD = 0.0
    z0 = 0.0
    b = 6.0

    rho0 = solve_observer_rho(M, MD, b, z0=z0)
    print(rho0)

    alfa = np.linspace(-np.arctan(10 / 15), np.arctan(10 / 15), 752)
    beta = np.linspace(np.arctan(10 / 15), -np.arctan(10 / 15), 752)

    Mat = np.loadtxt("Mat")
    Mz = np.loadtxt("Mz")

    M2 = classify_shadow_disk(Mat, Mz, b)
    Msym = mirror_quadrants(M2)

    count = int(np.sum(Msym == 1))
    print(count)

    end = time.time()
    print(end - start)

    plt.figure(figsize=(15, 15))
    cm = get_colormap("darkgrey_black")
    plt.imshow(Msym, cmap=cm, extent=[-beta[0], -beta[-1], alfa[0], alfa[-1]])
    plt.title("$M_{BN}=$" + str(M) + " , $M_{Disk} =$" + str(MD), size=40)
    plt.xlabel("$\\beta$", size=30)
    plt.ylabel("$\\alpha$", size=30)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.savefig("Test")
