#!/usr/bin/env python
"""Serial Weyl-family shadow ray tracer with live 3-colour plotting.

Two modes are available (see weyl_core module docstring for the `m` selector convention):

  * BH only:     MD = 0.0  -> load a matrix tabulated with MD = 0.0 (e.g. "Mat_nu_disk0.0")
  * BH + disk:    MD > 0.0  -> load a matrix tabulated with the matching MD

This driver defaults to the BH-only mode (MD = 0.0), matching the historical
test_Z_SHADOW.py behaviour. `generate_matriz.py` must be run first to produce the
matrix file for the chosen (M, MD, b).
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import weyl_core as wc
from weyl_plot import classify_shadow_disk, get_colormap

if __name__ == "__main__":
    # --- select spacetime mode here -------------------------------------------------
    M = 1.0
    MD = 0.0  # MD = 0.0 -> black-hole-only mode (default here) ; MD > 0.0 -> BH + disk
    z0 = 0.0
    b = 6.0
    hder = 10**-6

    wc.load_matrix("Mat_nu_disk0.0")  # must match MD above; load BEFORE first trace

    rho0 = wc.solve_observer_rho(M, MD, b, z0=z0)
    print(rho0)

    alfaa = np.linspace(-np.arctan(10 / 15), np.arctan(10 / 15), 80)
    betaa = np.linspace(np.arctan(10 / 15), -np.arctan(10 / 15), 80)

    alfa = np.linspace(alfaa[0], alfaa[int(len(alfaa) / 2) - 1], int(len(alfaa) / 2))
    beta = np.linspace(betaa[0], betaa[int(len(betaa) / 2) - 1], int(len(betaa) / 2))

    Mat = np.zeros((len(alfa), len(beta)))
    Mz = np.zeros((len(alfa), len(beta)))

    start = time.time()
    for i in range(len(alfa)):
        for j in range(len(beta)):
            y = np.array(
                [
                    rho0,
                    wc.dr(rho0, z0, M, MD, b, alfa[i], beta[j]),
                    z0,
                    wc.dthe(rho0, z0, M, MD, b, alfa[i]),
                ]
            )
            (Mat[i, j], Mz[i, j]) = wc.func_bh_disk_crossing(
                y, 300.0, -0.02, alfa[i], beta[j], M, rho0, z0, MD, b, hder
            )
    end = time.time()
    print(end - start)

    M2 = classify_shadow_disk(Mat, Mz, b)

    plt.figure(figsize=(15, 15))
    cm = get_colormap("white_black_grey")
    plt.imshow(M2, cmap=cm, extent=[-beta[0], -beta[-1], alfa[0], alfa[-1]])
    plt.xlabel("$\\beta$", size=30)
    plt.ylabel("$\\alpha$", size=30)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.show()
