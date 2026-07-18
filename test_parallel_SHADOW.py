#!/usr/bin/env python
"""Production Weyl-family shadow ray tracer: numba `prange` multicore, via `f_paral`.

Two modes are available, selected by the disk mass `MD` and the matching lookup matrix
(see weyl_core module docstring for the `m` selector convention):

  * BH only:      MD = 0.0  -> load a matrix tabulated with MD = 0.0 (e.g. "Mat_nu_disk0.0")
  * BH + disk:     MD > 0.0  -> load a matrix tabulated with the matching MD (e.g. "Mat_nu_disk0.1")

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

    alfaa = np.linspace(-np.arctan(10 / 15), np.arctan(10 / 15), 160)
    betaa = np.linspace(np.arctan(10 / 15), -np.arctan(10 / 15), 160)

    alfa = np.linspace(alfaa[0], alfaa[int(len(alfaa) / 2) - 1], int(len(alfaa) / 2))
    beta = np.linspace(betaa[0], betaa[int(len(betaa) / 2) - 1], int(len(betaa) / 2))

    start1 = time.time()
    start = time.time()
    (Mat, Mz) = wc.f_paral(rho0, z0, M, MD, b, alfa, beta, hder)
    end = time.time()
    print(end - start)

    np.savetxt("Mat", Mat)
    np.savetxt("Mz", Mz)

    end1 = time.time()
    print(end1 - start1)
