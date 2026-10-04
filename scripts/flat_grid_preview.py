"""Render the end-z grid for straight (unlensed) rays, as in flat Euclidean space.

The observer sits at x = d on the x axis, phi = 0. Each pixel ray goes straight (backwards in time,
towards and past the centre) and ends on the sphere |p| = 30. The y and z signs follow the traced
convention (checked against a real run at the corner pixels). Same layout and plot code as a
real run, so the picture shows the grid before the black hole bends it.

Usage: uv run python scripts/flat_grid_preview.py configs/<cfg>.toml OUT.png [--n 1024]
"""

import argparse

import numpy as np

from bh_shadows.plots import visuals as vis
from bh_shadows.runner import run_shadow as rs
from bh_shadows.runner import shadow_config as sc

R_ESC = 30.0


def flat_run(cfg: sc.ShadowConfig, z_sign: float = -1.0) -> dict[str, np.ndarray]:
    """Return a run dict for straight rays (all escaped).

    Parameters
    ----------
    cfg : ShadowConfig
        Run config (uses `n`, `half_angle`, `distance`).
    z_sign : float, optional
        Sign of the end z (the traced convention is fixed by a comparison, see `main`).

    Returns
    -------
    dict of str to np.ndarray
        `status`, `coord1` (rho), `coord2` (z) and `phi`, each of shape (n, n).
    """
    alpha, beta = rs._angles(cfg)
    a, b = np.meshgrid(alpha, beta, indexing="ij")
    d = cfg.distance
    ux, uy, uz = -np.cos(a) * np.cos(b), -np.cos(a) * np.sin(b), np.sin(a)
    # |p0 + t u| = R: t^2 + 2 d ux t + d^2 - R^2 = 0, positive root
    t = -d * ux + np.sqrt((d * ux) ** 2 - d**2 + R_ESC**2)
    x, y, z = d + t * ux, t * uy, t * uz
    return {
        "status": np.zeros(a.shape, int),
        "coord1": np.hypot(x, y),
        "coord2": z_sign * z,
        "phi": np.arctan2(y, x),
    }


def main() -> None:
    """Render the flat-space grid."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("out")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--z-sign", type=float, default=-1.0)
    args = p.parse_args()
    cfg = sc.ShadowConfig.from_toml(args.config, {"n": args.n, "show": False})
    vis.render_image(flat_run(cfg, args.z_sign), cfg, args.out)


if __name__ == "__main__":
    main()
