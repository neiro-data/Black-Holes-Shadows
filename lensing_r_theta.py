#!/usr/bin/env python
"""Kerr-metric (r, theta) shadow/lensing driver: thin wrapper around kerr_core.

Self-contained (no lambda-potential lookup matrix needed, unlike the Weyl family).
"""

import time

import numpy as np

import kerr_core as kc

if __name__ == "__main__":
    start = time.time()
    a = 0.0
    M = 1.0
    r0 = 15.0
    theta0 = np.pi / 2

    alfaa = np.linspace(-np.arctan(10 / 15), np.arctan(10 / 15), 752)
    betaa = np.linspace(np.arctan(10 / 15), -np.arctan(10 / 15), 752)

    alfa = np.linspace(alfaa[0], alfaa[int(len(alfaa) / 2) - 1], int(len(alfaa) / 2))
    beta = np.linspace(betaa[0], betaa[int(len(betaa) / 2) - 1], int(len(betaa) / 2))

    Mat = np.zeros((len(alfa), len(beta)))
    Mthet = np.zeros((len(alfa), len(beta)))
    Mphi = np.zeros((len(alfa), len(beta)))
    for i in range(len(alfa)):
        for j in range(len(beta)):
            y = np.array(
                [
                    r0,
                    kc.dr(r0, a, M, theta0, alfa[i], beta[j]),
                    theta0,
                    kc.dthe(r0, a, M, theta0, alfa[i]),
                    0,
                ]
            )
            (Mat[i, j], Mthet[i, j], Mphi[i, j]) = kc.func(
                y, 300.0, -0.02, alfa[i], beta[j], a, M, r0, theta0
            )
    end = time.time()
    print(end - start)

    np.savetxt("Mat", Mat)
    np.savetxt("Mthet", Mthet)
    np.savetxt("Mphi", Mphi)

    # Analytic photon-ring / lensed-image overlay helpers available for further plotting:
    # kc.r1, kc.r2, kc.yplus, kc.yminus, kc.x (see kerr_core module for definitions).
