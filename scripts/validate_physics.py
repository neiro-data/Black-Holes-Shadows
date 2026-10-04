#!/usr/bin/env python
"""Physics-validation harness: trace one small grid and print shadow metrics as JSON.

    uv run python scripts/validate_physics.py --config configs/schwarzschild_nodisk.toml --n 64

One config per process (`weyl_core.Mat_nu` is frozen at first numba compile). n is capped at 64.
`--raw-kerr-a` traces `kerr_core.f_paral` directly, bypassing `ShadowConfig` (which forbids a < 0),
and optionally dumps the status grid with `--save-status`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

N_MAX = 64


def edge_metrics(status: np.ndarray, h: float) -> dict:
    """Central-line shadow edges (midpoint of captured/escaped pixels), centroid, half-widths."""
    n = status.shape[0]
    ang = np.linspace(-h, h, n)  # alpha rows; also x = -beta columns
    pitch = 2 * h / (n - 1)
    c = (n // 2 - 1, n // 2)
    out: dict = {"pitch": pitch}

    def span(line: np.ndarray) -> tuple[float, float] | None:
        idx = np.nonzero(line == 1)[0]
        if idx.size == 0 or idx.min() == 0 or idx.max() == n - 1:
            return None
        lo, hi = idx.min(), idx.max()
        return (ang[lo] + ang[lo - 1]) / 2, (ang[hi] + ang[hi + 1]) / 2

    for key, lines in (("alfa", [status[:, i] for i in c]), ("x", [status[i, :] for i in c])):
        spans = [span(ln) for ln in lines]
        if None in spans:
            out[f"{key}_lo"] = out[f"{key}_hi"] = None
        else:
            out[f"{key}_lo"] = float(np.mean([s[0] for s in spans]))
            out[f"{key}_hi"] = float(np.mean([s[1] for s in spans]))
    out["alfa_edge_top"], out["alfa_edge_bottom"] = out.pop("alfa_hi"), out.pop("alfa_lo")
    out["beta_edge_left"], out["beta_edge_right"] = out.pop("x_lo"), out.pop("x_hi")
    shadow = status == 1
    out["centroid_x"] = float(np.nonzero(shadow)[1].mean() * pitch - h) if shadow.any() else None
    out["centroid_alfa"] = float(np.nonzero(shadow)[0].mean() * pitch - h) if shadow.any() else None
    if out["beta_edge_left"] is not None:
        out["halfwidth_left"] = -out["beta_edge_left"]
        out["halfwidth_right"] = out["beta_edge_right"]
    return out


def trace(args) -> tuple[np.ndarray, float]:
    from shadow_config import ShadowConfig

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
        import kerr_core as kc
        from run_shadow import _angles

        alfa, beta = _angles(cfg)
        *_, st = kc.f_paral(
            cfg.distance, np.deg2rad(cfg.theta0), args.raw_kerr_a, cfg.M, alfa, beta
        )
        return np.asarray(st), cfg.half_angle
    import generate_matriz as gm
    import run_shadow as rs

    if args.matrices_dir:
        gm.MATRICES_DIR = Path(args.matrices_dir)
    rs.limit_threads()
    return rs.trace_grid(cfg, progress=False)["status"], cfg.half_angle


def main(argv=None) -> None:
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
