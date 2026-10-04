#!/usr/bin/env python
"""Trace one small grid and print the shadow metrics as JSON.

    uv run python scripts/validate_physics.py --config configs/schwarzschild_nodisk.toml --n 64

Use one config for each process. Numba fixes `weyl_core.Mat_nu` at the first compile. The limit
is n = 64. `--raw-kerr-a` calls `kerr_core.f_paral` directly, without `ShadowConfig` (which does
not accept a < 0). `--save-status` writes the status image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

N_MAX = 64


def edge_metrics(status: np.ndarray, h: float) -> dict:
    """Measure the shadow edges on the two central lines.

    An edge is the midpoint between a captured and an escaped pixel.

    Parameters
    ----------
    status : np.ndarray
        Status image of shape (n, n).
    h : float
        Half angle of the field of view (radians).

    Returns
    -------
    dict
        Pixel pitch, alpha and beta edges, centroid and half widths (radians). An edge is None
        if the line has no shadow.
    """
    n = status.shape[0]
    ang = np.linspace(-h, h, n)  # alpha rows; also x = -beta columns
    pitch = 2 * h / (n - 1)
    c = (n // 2 - 1, n // 2)
    out: dict = {"pitch": pitch}

    def span(line: np.ndarray) -> tuple[float, float] | None:
        """Return the shadow edges `(lo, hi)` on one line, or None (no shadow, or at the border)."""
        idx = np.nonzero(line == 1)[0]
        if idx.size == 0 or idx.min() == 0 or idx.max() == n - 1:
            return None
        lo, hi = idx.min(), idx.max()
        return (ang[lo] + ang[lo - 1]) / 2, (ang[hi] + ang[hi + 1]) / 2

    for key, lines in (("alpha", [status[:, i] for i in c]), ("x", [status[i, :] for i in c])):
        spans = [span(ln) for ln in lines]
        if None in spans:
            out[f"{key}_lo"] = out[f"{key}_hi"] = None
        else:
            out[f"{key}_lo"] = float(np.mean([s[0] for s in spans]))
            out[f"{key}_hi"] = float(np.mean([s[1] for s in spans]))
    out["alpha_edge_top"], out["alpha_edge_bottom"] = out.pop("alpha_hi"), out.pop("alpha_lo")
    out["beta_edge_left"], out["beta_edge_right"] = out.pop("x_lo"), out.pop("x_hi")
    shadow = status == 1
    out["centroid_x"] = float(np.nonzero(shadow)[1].mean() * pitch - h) if shadow.any() else None
    out["centroid_alpha"] = (
        float(np.nonzero(shadow)[0].mean() * pitch - h) if shadow.any() else None
    )
    if out["beta_edge_left"] is not None:
        out["halfwidth_left"] = -out["beta_edge_left"]
        out["halfwidth_right"] = out["beta_edge_right"]
    return out


def trace(args) -> tuple[np.ndarray, float]:
    """Trace one grid from the command-line options.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments. With `raw_kerr_a`, the function calls `kerr_core.f_paral` directly.

    Returns
    -------
    status : np.ndarray
        Status image.
    h : float
        Half angle of the field of view (radians).
    """
    from bh_shadows.runner.shadow_config import ShadowConfig

    ov = {
        "n": args.n,
        "n_grid": args.n_grid,
        "M": args.M,
        "MD": args.MD,
        "a": 0.0 if args.raw_kerr_a is not None else args.a,
        "theta0": args.theta0,
        "disk_visible_opt": {"true": True, "false": False, "auto": "auto"}[args.disk_visible],
        "backend_opt": args.backend_opt,
        "save": False,
        "show": False,
    }
    if args.raw_kerr_a is not None:
        ov["backend_opt"] = "kerr"
    cfg = ShadowConfig.from_toml(args.config, ov)
    if args.raw_kerr_a is not None:
        from bh_shadows.core import kerr_core as kc
        from bh_shadows.runner.run_shadow import _angles

        alpha, beta = _angles(cfg)
        *_, st = kc.f_paral(
            cfg.distance, np.deg2rad(cfg.theta0), args.raw_kerr_a, cfg.M, alpha, beta
        )
        return np.asarray(st), cfg.half_angle
    from bh_shadows.core import matrix as gm
    from bh_shadows.runner import run_shadow as rs

    if args.matrices_dir:
        gm.MATRICES_DIR = Path(args.matrices_dir)
    rs.limit_threads()
    return rs.trace_grid(cfg, progress=False)["status"], cfg.half_angle


def main(argv=None) -> None:
    """Trace one grid and print the status counts and edge metrics as JSON.

    Parameters
    ----------
    argv : list of str, optional
        Arguments. The default is `sys.argv[1:]`.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs/schwarzschild_nodisk.toml"))
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--M", type=float)
    p.add_argument("--MD", type=float)
    p.add_argument("--a", type=float)
    p.add_argument("--theta0", type=float)
    p.add_argument("--disk-visible", choices=["auto", "true", "false"], default="auto")
    p.add_argument("--backend", choices=["auto", "weyl", "kerr"], dest="backend_opt")
    p.add_argument("--n-grid", type=int)
    p.add_argument("--matrices-dir")
    p.add_argument("--raw-kerr-a", type=float, help="trace kerr_core directly (allows a < 0)")
    p.add_argument("--save-status", help="write the status grid to this .npy path")
    args = p.parse_args(argv)
    if not 0 < args.n <= N_MAX:
        p.error(f"--n must satisfy 0 < n <= {N_MAX}, got {args.n}")
    status, h = trace(args)
    if args.save_status:
        np.save(args.save_status, status)
    counts = {str(s): int((status == s).sum()) for s in range(4)}
    print(json.dumps({"n": args.n, "counts": counts, **edge_metrics(status, h)}))


if __name__ == "__main__":
    main()
