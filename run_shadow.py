#!/usr/bin/env python
"""Config-driven end-to-end shadow runner: one config in, one traced + plotted shadow out.

    uv run python run_shadow.py configs/schwarzschild_disk.toml --MD 0.0 --b 6.0

One config per process: `weyl_core.Mat_nu` is frozen at first numba compile.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import weyl_core as wc
from generate_matriz import ensure_matrix
from shadow_config import ShadowConfig
from weyl_plot import classify_shadow_disk, mirror_quadrants, plot_status

HDER = 10**-6


def _angles(cfg: ShadowConfig) -> tuple[np.ndarray, np.ndarray]:
    h = cfg.half_angle
    return np.linspace(-h, h, cfg.n), np.linspace(h, -h, cfg.n)


def _mirror4(q: np.ndarray, sign_flip_lower: bool = False) -> np.ndarray:
    full = mirror_quadrants(q)
    if sign_flip_lower:
        full[q.shape[0] :, :] *= -1
    return full


def _trace_weyl(cfg: ShadowConfig) -> dict[str, np.ndarray]:
    """Trace one quadrant (static, equatorially symmetric) and mirror both axes."""
    mpath = ensure_matrix(cfg.M, cfg.MD, cfg.b, cfg.n_grid, cfg.matrix_source)
    wc.load_matrix(str(mpath))  # must precede the first trace
    z0 = 0.0
    rho0 = wc.solve_observer_rho(cfg.M, cfg.MD, cfg.b, z0=z0, target_gpp_sqrt=cfg.distance)

    alfa_f, beta_f = _angles(cfg)
    half = cfg.n // 2
    alfa, beta = alfa_f[:half], beta_f[:half]
    if cfg.parallel:
        Mat, Mz = wc.f_paral(rho0, z0, cfg.M, cfg.MD, cfg.b, alfa, beta, HDER, cfg.disk_visible)
    else:
        Mat, Mz = np.zeros((half, half)), np.zeros((half, half))
        for i in range(half):
            for j in range(half):
                y = np.array(
                    [
                        rho0,
                        wc.dr(rho0, z0, cfg.M, cfg.MD, cfg.b, alfa[i], beta[j]),
                        z0,
                        wc.dthe(rho0, z0, cfg.M, cfg.MD, cfg.b, alfa[i]),
                    ]
                )
                Mat[i, j], Mz[i, j] = wc.func(
                    y,
                    300.0,
                    -0.02,
                    alfa[i],
                    beta[j],
                    cfg.M,
                    rho0,
                    z0,
                    cfg.MD,
                    cfg.b,
                    HDER,
                    cfg.disk_visible,
                )

    status = classify_shadow_disk(Mat, Mz, cfg.b)
    status[np.isnan(Mat) | np.isnan(Mz)] = 3
    return {
        "status": _mirror4(status).astype(int),
        "coord1": _mirror4(Mat),
        "coord2": _mirror4(Mz, sign_flip_lower=True),
    }


def _trace_kerr(cfg: ShadowConfig) -> dict[str, np.ndarray]:
    """Trace the full beta range (frame dragging breaks left/right symmetry).

    Alpha is mirrored only at theta0 == 90 deg; otherwise the whole grid is traced.
    """
    import kerr_core as kc

    theta0 = np.deg2rad(cfg.theta0)
    alfa, beta = _angles(cfg)
    mirror = cfg.theta0 == 90.0
    alfa_t = alfa[: cfg.n // 2] if mirror else alfa

    if cfg.parallel:
        Mr, Mth, Mphi, Mst = kc.f_paral(cfg.distance, theta0, cfg.a, cfg.M, alfa_t, beta)
    else:
        Mr, Mth, Mphi = (np.zeros((len(alfa_t), len(beta))) for _ in range(3))
        Mst = np.zeros((len(alfa_t), len(beta)), dtype=int)
        for i in range(len(alfa_t)):
            for j in range(len(beta)):
                y = np.array(
                    [
                        cfg.distance,
                        kc.dr(cfg.distance, cfg.a, cfg.M, theta0, alfa_t[i], beta[j]),
                        theta0,
                        kc.dthe(cfg.distance, cfg.a, cfg.M, theta0, alfa_t[i]),
                        0.0,
                    ]
                )
                Mr[i, j], Mth[i, j], Mphi[i, j], Mst[i, j] = kc.func_status(
                    y, 300.0, -0.02, alfa_t[i], beta[j], cfg.a, cfg.M, cfg.distance, theta0
                )

    out = {"status": Mst, "coord1": Mr, "coord2": Mth, "phi": Mphi}
    if mirror:  # equatorial reflection: theta -> pi - theta, r and phi unchanged
        out = {k: np.vstack([v, v[::-1]]) for k, v in out.items()}
        out["coord2"][len(alfa_t) :] = np.pi - out["coord2"][len(alfa_t) :]
    return out


def trace_grid(cfg: ShadowConfig) -> dict[str, np.ndarray]:
    """Trace the full image for `cfg` and return status/coord1/coord2 (and phi for Kerr)."""
    return _trace_weyl(cfg) if cfg.backend == "weyl" else _trace_kerr(cfg)


def save_run(cfg: ShadowConfig, out: dict[str, np.ndarray]) -> None:
    run_dir = cfg.run_dir
    cfg.write_resolved()
    np.savetxt(run_dir / "status.txt", out["status"], fmt="%d")
    for key in ("coord1", "coord2", "phi"):
        if key in out:
            np.savetxt(run_dir / f"{key}.txt", out[key])
    if cfg.save or cfg.show:
        plot_status(out["status"], cfg, run_dir / "image.png" if cfg.save else None)


def parse_args(argv=None) -> tuple[str, dict]:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--M", type=float)
    p.add_argument("--a", type=float)
    p.add_argument("--MD", type=float)
    p.add_argument("--b", type=float)
    p.add_argument("--n", type=int)
    p.add_argument("--name")
    p.add_argument("--disk-visible", choices=["auto", "true", "false"], dest="disk_visible_opt")
    args = vars(p.parse_args(argv))
    cfg_path = args.pop("config")
    if args.get("disk_visible_opt") in ("true", "false"):
        args["disk_visible_opt"] = args["disk_visible_opt"] == "true"
    return cfg_path, {k: v for k, v in args.items() if v is not None}


def main(argv=None) -> None:
    cfg_path, overrides = parse_args(argv)
    cfg = ShadowConfig.from_toml(cfg_path, overrides)
    print(f"backend={cfg.backend} disk_visible={cfg.disk_visible} run_dir={cfg.run_dir}")
    t0 = time.time()
    out = trace_grid(cfg)
    print(f"traced in {time.time() - t0:.1f}s; shadow pixels: {int((out['status'] == 1).sum())}")
    save_run(cfg, out)


if __name__ == "__main__":
    main()
