#!/usr/bin/env python
"""Build the lambda look-up table for the Weyl tracers.

`build_lambda_matrix` is the only implementation. The runner calls `ensure_matrix`. That function
keeps one table for each parameter set in `matrices/`, with a JSON sidecar that holds the
parameters.

    uv run bh-matrix --M 0.9 --MD 0.1 --b 6.0
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from bh_shadows.core.weyl_core import lamb_Mat

log = logging.getLogger(__name__)

MATRICES_DIR = Path("matrices")
EXTENT = 40.0  # the grid spans z in [40, -40] and rho in [40, 0]


def _lamb_sch_grid(rho: np.ndarray, z: np.ndarray, M: float) -> np.ndarray:
    """Calculate the analytic Schwarzschild lambda on a grid (vector form of `lambSch`).

    Parameters
    ----------
    rho, z : np.ndarray
        Weyl coordinates. The arrays must broadcast together.
    M : float
        Black-hole mass.

    Returns
    -------
    np.ndarray
        Lambda values. Points on the axis segment give NaN or -inf.
    """
    d1 = np.sqrt(rho**2 + (z - M) ** 2)
    d2 = np.sqrt(rho**2 + (z + M) ** 2)
    sigma = np.sqrt((rho**2 + z**2 + M**2) ** 2 - 4 * z**2 * M**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(((d1 + d2) ** 2 - 4 * M**2) / (4 * sigma))


def build_lambda_matrix(
    M: float, MD: float, b: float, n_grid: int, source: str = "auto"
) -> np.ndarray:
    """Build the lambda look-up table.

    Parameters
    ----------
    M, MD, b : float
        Black-hole mass, disk mass and disk inner radius.
    n_grid : int
        Number of grid points on each axis.
    source : {"auto", "analytic", "quadrature"}, optional
        Build method. "auto" uses "analytic" when `MD == 0`, else "quadrature".

    Returns
    -------
    np.ndarray
        Table of shape (n_grid, n_grid). Rows go over z from 40 to -40. Columns go over rho from
        40 to 0. NaN values are set to -20.

    Raises
    ------
    ValueError
        If `source` is unknown, or if `source="analytic"` and `MD != 0`.
    """
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
    """Return the cache path of one lambda table.

    Parameters
    ----------
    M, MD, b : float
        Black-hole mass, disk mass and disk inner radius.
    n_grid : int
        Number of grid points on each axis.

    Returns
    -------
    Path
        `MATRICES_DIR / "lambda_M{M}_MD{MD}_b{b}_n{n_grid}.npz"`.
    """
    return MATRICES_DIR / f"lambda_M{M}_MD{MD}_b{b}_n{n_grid}.npz"


def ensure_matrix(M: float, MD: float, b: float, n_grid: int, source: str = "auto") -> Path:
    """Return the path of a cached lambda table. Build the table if it does not exist.

    Parameters
    ----------
    M, MD, b : float
        Black-hole mass, disk mass and disk inner radius.
    n_grid : int
        Number of grid points on each axis.
    source : {"auto", "analytic", "quadrature"}, optional
        Build method if the table does not exist.

    Returns
    -------
    Path
        Path of the `.npz` file. A `.json` sidecar beside it holds the parameters.

    Raises
    ------
    ValueError
        If the file exists but its sidecar is missing or has other parameters.
    """
    path = matrix_path(M, MD, b, n_grid)
    sidecar = path.with_suffix(".json")
    params = {"M": M, "MD": MD, "b": b, "n_grid": n_grid}
    if path.exists():
        if not sidecar.exists() or json.loads(sidecar.read_text()) != params:
            raise ValueError(f"{path} does not match the requested parameters {params}")
        log.info("lambda matrix cached, loading %s", path)
        return path
    if source == "auto":
        source = "analytic" if MD == 0 else "quadrature"
    log.info(
        "building lambda matrix M=%s MD=%s b=%s n_grid=%d source=%s -> %s",
        M,
        MD,
        b,
        n_grid,
        source,
        path,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    lam = build_lambda_matrix(M, MD, b, n_grid, source)
    np.savez_compressed(path, lam=lam)
    log.info("lambda matrix built in %.1fs, saved %s", time.perf_counter() - t0, path)
    sidecar.write_text(json.dumps(params))
    return path


def main() -> None:
    """Build or reuse one lambda table from the command line, then print its path."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--M", type=float, default=1.0)
    p.add_argument("--MD", type=float, default=0.0)
    p.add_argument("--b", type=float, default=6.0)
    p.add_argument("--n-grid", type=int, default=1200)
    p.add_argument("--source", default="auto", choices=["auto", "analytic", "quadrature"])
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    print(ensure_matrix(args.M, args.MD, args.b, args.n_grid, args.source))


if __name__ == "__main__":
    main()
