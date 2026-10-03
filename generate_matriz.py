#!/usr/bin/env python
"""Tabulate the non-linear lambda-potential lookup matrix for the Weyl-family ray tracers.

`build_lambda_matrix` is the single implementation. The runner (`run_shadow.py`) calls
`ensure_matrix`, which caches by content under `matrices/` with a JSON sidecar holding the
parameters. The `__main__` block is a thin CLI over the same functions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from weyl_core import lamb_Mat

MATRICES_DIR = Path("matrices")
EXTENT = 40.0  # the grid spans z in [40, -40] and rho in [40, 0]


def _lamb_sch_grid(rho: np.ndarray, z: np.ndarray, M: float) -> np.ndarray:
    """Vectorised copy of `weyl_core.lambSch` (the analytic lambda for MD = 0)."""
    d1 = np.sqrt(rho**2 + (z - M) ** 2)
    d2 = np.sqrt(rho**2 + (z + M) ** 2)
    sigma = np.sqrt((rho**2 + z**2 + M**2) ** 2 - 4 * z**2 * M**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(((d1 + d2) ** 2 - 4 * M**2) / (4 * sigma))


def build_lambda_matrix(
    M: float, MD: float, b: float, n_grid: int, source: str = "auto"
) -> np.ndarray:
    """Return the (n_grid x n_grid) lambda table, rows over z (40 -> -40), columns over rho."""
    if source == "auto":
        source = "analytic" if MD == 0 else "quadrature"
    z = np.linspace(EXTENT, -EXTENT, n_grid)
    rho = np.linspace(EXTENT, 0.0, n_grid)

    if source == "analytic":
        if MD != 0:
            raise ValueError("the analytic lambda table is only valid for MD == 0")
        mat = _lamb_sch_grid(rho[None, :], z[:, None], M)
    elif source == "quadrature":
        mat = np.zeros((n_grid, n_grid))
        for i in range(n_grid):
            for j in range(n_grid):
                # m=2 (BH + disk) is the selector the geodesic code always uses. With MD = 0 it
                # is identical to m=0, so no physics change.
                mat[i, j] = lamb_Mat(rho[j], z[i], M, MD, b, 2, 10**-6)
    else:
        raise ValueError(f"unknown matrix source {source!r}")

    mat[np.isnan(mat)] = -20
    return mat


def matrix_path(M: float, MD: float, b: float, n_grid: int) -> Path:
    return MATRICES_DIR / f"lambda_M{M}_MD{MD}_b{b}_n{n_grid}.txt"


def ensure_matrix(M: float, MD: float, b: float, n_grid: int, source: str = "auto") -> Path:
    """Return the cached matrix path, building it if missing. Refuse a sidecar mismatch."""
    path = matrix_path(M, MD, b, n_grid)
    sidecar = path.with_suffix(".json")
    params = {"M": M, "MD": MD, "b": b, "n_grid": n_grid}
    if path.exists():
        if not sidecar.exists() or json.loads(sidecar.read_text()) != params:
            raise ValueError(f"{path} does not match the requested parameters {params}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, build_lambda_matrix(M, MD, b, n_grid, source))
    sidecar.write_text(json.dumps(params))
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--M", type=float, default=1.0)
    p.add_argument("--MD", type=float, default=0.0)
    p.add_argument("--b", type=float, default=6.0)
    p.add_argument("--n-grid", type=int, default=1200)
    p.add_argument("--source", default="auto", choices=["auto", "analytic", "quadrature"])
    args = p.parse_args()
    print(ensure_matrix(args.M, args.MD, args.b, args.n_grid, args.source))
