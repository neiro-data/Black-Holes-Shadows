"""Analytic shadow-edge curves in plot units (x = -beta axis, y = alpha axis, radians)."""

from __future__ import annotations

import numpy as np

import kerr_core as kc

A_ZERO_TOL = 1e-12


def schwarzschild_edge_angle(M: float, r0: float) -> float:
    """Angular radius of the Schwarzschild shadow seen from a static observer at `r0`."""
    return float(np.arcsin(3 * np.sqrt(3) * M / r0 * np.sqrt(1 - 2 * M / r0)))


def kerr_shadow_edge(
    a: float, M: float, r0: float, theta0: float, n: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    """Closed shadow-edge curve `(x_angle, y_angle)`; `theta0` in radians.

    For |a| < 1e-12 this is the Schwarzschild circle (the Kerr helpers divide by a).
    """
    if abs(a) < A_ZERO_TOL:
        t = np.linspace(0.0, 2 * np.pi, n)
        rad = schwarzschild_edge_angle(M, r0)
        return rad * np.cos(t), rad * np.sin(t)
    lo, hi = sorted((float(kc.r1(a, M)), float(kc.r2(a, M))))
    r = np.linspace(lo, hi, n)
    scale = np.sqrt(kc.gpp(r0, a, M, theta0))
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = (kc.eta(r, a, M) >= 0) & np.isfinite(kc.x(r, a, M, theta0, r0))
        r = r[ok]
        xs = kc.x(r, a, M, theta0, r0) / scale
        yp = kc.yplus(r, a, M, theta0, r0) / scale
        ym = kc.yminus(r, a, M, theta0, r0) / scale
    good = np.isfinite(xs) & np.isfinite(yp) & np.isfinite(ym)
    xs, yp, ym = xs[good], yp[good], ym[good]
    xc = np.concatenate([xs, xs[::-1], xs[:1]])
    yc = np.concatenate([yp, ym[::-1], yp[:1]])
    return xc, yc


def config_edge(cfg) -> tuple[np.ndarray, np.ndarray] | None:
    """Analytic edge for a `ShadowConfig`, or None when none exists (Weyl with a disk)."""
    if cfg.MD > 0:
        return None
    return kerr_shadow_edge(cfg.a, cfg.M, cfg.distance, np.deg2rad(cfg.theta0))


def draw_edge(ax, cfg) -> None:
    """Draw the analytic shadow edge on `ax` when `cfg.shadow_edge` is set and a curve exists."""
    if not cfg.shadow_edge:
        return
    curve = config_edge(cfg)
    if curve is not None:
        ax.plot(*curve, color="cyan", linestyle="--", linewidth=1.0)
