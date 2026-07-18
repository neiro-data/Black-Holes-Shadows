#!/usr/bin/env python
"""Prerequisite pipeline step: tabulate the non-linear lambda-potential lookup matrix.

Must be run before any Weyl-family ray tracer (test_parallel_SHADOW.py, test_Z_SHADOW.py,
test_symmetry_lensing.py, symmetry.py) that loads a matrix via `weyl_core.load_matrix`.

Saves e.g. "Mat_constA_Mbh_0.9" via `np.savetxt` for the given (M, MD, b) configuration.
"""

import numpy as np

from weyl_core import lamb_Mat, solve_observer_rho

if __name__ == "__main__":
    z = np.linspace(40.0, -40.0, 1200)
    rho = np.linspace(40.0, 0.0, 1200)
    nu_Mat = np.zeros((len(z), len(rho)))

    # BH + disk configuration this matrix is tabulated for. m=2 (BH+disk) is baked into
    # lamb_Mat's use of lambSch; see weyl_core module docstring for the `m` selector.
    M = 0.9
    MD = 0.1
    z0 = 0.0
    b = 3.0
    rho0 = solve_observer_rho(M, MD, b, z0=z0)

    count = 0
    for i in range(len(z)):
        for j in range(len(rho)):
            nu_Mat[i, j] = lamb_Mat(rho[j], z[i], M, MD, b, 2, 10**-6)

    for i in range(len(z)):
        for j in range(len(rho)):
            if np.isnan(nu_Mat[i, j]):
                nu_Mat[i, j] = -20
                count += 1
                print(z[i], rho[j])

    print(count)
    np.savetxt("Mat_constA_Mbh_0.9", nu_Mat)
