#!/usr/bin/env python
"""Weyl-family lensing driver: traces rays with phi tracking, saves Mat/Mz/Mphi.

Two modes are available (see weyl_core module docstring for the `m` selector convention):

  * BH only:    MD = 0.0  -> matching matrix e.g. "Mat_nu_disk0.0"
  * BH + disk:   MD > 0.0  -> matching matrix e.g. "Mat_nu_disk0.1"

`generate_matriz.py` must be run first to produce the matrix file for the chosen (M, MD, b).
"""

import time

import numpy as np

import weyl_core as wc

if __name__ == "__main__":
    # --- select spacetime mode here -------------------------------------------------
    M = 0.9
    MD = 0.1  # MD = 0.0 -> black-hole-only mode ; MD > 0.0 -> black hole + disk mode
    z0 = 0.0
    b = 6.0
    hder = 10**-6

    wc.load_matrix("Mat_nu_disk0.1")  # must match MD above; load BEFORE first trace

    rho0 = wc.solve_observer_rho(M, MD, b, z0=z0)
    print(rho0)

    alfaa = np.linspace(-np.arctan(10 / 15), np.arctan(10 / 15), 1000)
    betaa = np.linspace(np.arctan(10 / 15), -np.arctan(10 / 15), 1000)

    alfa = np.linspace(alfaa[0], alfaa[int(len(alfaa) / 2) - 1], int(len(alfaa) / 2))
    beta = np.linspace(betaa[0], betaa[int(len(betaa) / 2) - 1], int(len(betaa) / 2))

    Mat = np.zeros((len(alfa), len(beta)))
    Mz = np.zeros((len(alfa), len(beta)))
    Mphi = np.zeros((len(alfa), len(beta)))

    start = time.time()
    for i in range(len(alfa)):
        for j in range(len(beta)):
            y = np.array(
                [
                    rho0,
                    wc.dr(rho0, z0, M, MD, b, alfa[i], beta[j]),
                    z0,
                    wc.dthe(rho0, z0, M, MD, b, alfa[i]),
                    0,
                ]
            )
            (Mat[i, j], Mz[i, j], Mphi[i, j]) = wc.func_lensing(
                y, 300.0, -0.02, alfa[i], beta[j], M, rho0, z0, MD, b, hder
            )
    end = time.time()
    print(end - start)

    np.savetxt("Mat", Mat)
    np.savetxt("Mz", Mz)
    np.savetxt("Mphi", Mphi)
