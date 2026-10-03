"""Selectable image visuals: colour each pixel from its ray outcome and escape direction.

Post-process only: it reads the arrays the tracers already return (no extra tracing). Pure numpy
except `render_image` and `load_sky_image` (matplotlib). `status` is 0 escaped, 1 captured,
2 disk, 3 failed. Escape angles `theta` in [0, pi], `phi` in [0, 2*pi) come from `sky_directions`:
Kerr uses `coord2` and `phi`; Weyl has no phi, so theta = pi - atan2(rho, z) (rows are
z-mirrored) and phi is a flat-sky approximation from the pixel column (beta).
Disk pixels have no direction behind the disk, so they blend
`disk_alpha * white` over the fixed neutral `DISK_BACKDROP`.

Light source (two_color / four_color, `light_source = true`): a white Gaussian glow around the
point behind the observer (`theta = pi - theta_obs`, `phi = pi`) is blended over the sky colour,
so lensed rays form a bright ring around the shadow. Layering: sky, glow, grid, then
captured/disk/failed.

Quadrants, as the observer sees them (flat-sky limit): upper-left red, upper-right green,
lower-left yellow, lower-right blue. "Upper" is theta < pi/2; left/right is the sign of
`phi_rel = wrap(phi - pi)` times `PHI_LEFT_SIGN`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from shadow_config import ShadowConfig

# Sign of phi_rel that means "left" in the image (fixed empirically against real runs).
PHI_LEFT_SIGN = 1

RED = np.array([0.91, 0.21, 0.16])
GREEN = np.array([0.47, 0.98, 0.33])
YELLOW = np.array([1.0, 1.0, 0.31])
BLUE = np.array([0.0, 0.0, 0.96])
GREY = np.array([0.505, 0.505, 0.505])
BLACK = np.zeros(3)
WHITE = np.ones(3)
GRID_COLOR = BLACK
DISK_BACKDROP = np.array([0.25, 0.25, 0.25])  # colour behind a semi-transparent disk

_TWO_PI = 2 * np.pi


@dataclass(frozen=True)
class VisualSpec:
    """Visual options for `compose_rgb` (mirrors the `[plot]` config keys)."""

    background: str = "four_color"
    grid: bool = False
    step_deg: float = 10.0
    width_deg: float = 0.4
    disk_alpha: float = 0.6
    sky: np.ndarray | None = None
    light_source: bool = False
    light_alpha: float = 0.7
    light_radius_deg: float = 10.0
    theta_obs: float = np.pi / 2  # observer polar angle (rad)

    @classmethod
    def from_config(cls, cfg) -> VisualSpec:
        """Build a spec from a `ShadowConfig`, loading the sky image when needed."""
        sky = load_sky_image(cfg.image_path) if cfg.background == "image" else None
        return cls(
            background=cfg.background,
            grid=cfg.grid,
            step_deg=cfg.grid_step_deg,
            width_deg=cfg.grid_width_deg,
            disk_alpha=cfg.disk_alpha,
            sky=sky,
            light_source=cfg.light_source,
            light_alpha=cfg.light_alpha,
            light_radius_deg=cfg.light_radius_deg,
            theta_obs=np.pi / 2 if cfg.backend == "weyl" else float(np.deg2rad(cfg.theta0)),
        )


def load_sky_image(path: str | Path) -> np.ndarray:
    """Read a PNG/JPEG as float RGB (H, W, 3) in [0, 1], dropping any alpha channel."""
    import matplotlib.pyplot as plt

    img = np.asarray(plt.imread(path))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3]
    if img.dtype == np.uint8:
        return img.astype(float) / 255.0
    return np.clip(img.astype(float), 0.0, 1.0)


def _phi_rel(phi: np.ndarray) -> np.ndarray:
    """Wrap phi - pi into (-pi, pi]."""
    return np.pi - np.mod(np.pi - (np.asarray(phi, dtype=float) - np.pi), _TWO_PI)


def sky_rgb(theta: np.ndarray, phi: np.ndarray, spec: VisualSpec) -> np.ndarray:
    """Sky colour for escape directions, shape `theta.shape + (3,)` (NaN angles give grey)."""
    theta = np.asarray(theta, dtype=float)
    rel = _phi_rel(phi)
    if spec.background == "greyscale":
        return np.broadcast_to(GREY, theta.shape + (3,)).copy()
    if spec.background == "image":
        if spec.sky is None:
            raise ValueError("background 'image' needs spec.sky")
        h, w = spec.sky.shape[:2]
        u = np.nan_to_num((rel + np.pi) / _TWO_PI)
        v = np.nan_to_num(theta / np.pi)
        col = np.clip((u * w).astype(int), 0, w - 1)
        row = np.clip((v * h).astype(int), 0, h - 1)
        return spec.sky[row, col]
    upper = theta < np.pi / 2
    if spec.background == "two_color":
        return np.where(upper[..., None], RED, BLUE)
    if spec.background == "four_color":
        left = PHI_LEFT_SIGN * rel > 0
        out = np.where(left[..., None], YELLOW, BLUE)
        return np.where(upper[..., None], np.where(left[..., None], RED, GREEN), out)
    raise ValueError(f"no sky for background {spec.background!r}")


def grid_mask(theta: np.ndarray, phi: np.ndarray, step_deg: float, width_deg: float) -> np.ndarray:
    """Boolean mask of celestial-sphere grid lines (theta and phi multiples of `step_deg`).

    Theta lines: `|theta - k*step| <= w`. Phi lines: `|dphi| * sin(theta) <= w`, which keeps the
    apparent line width even near the poles.
    """
    theta = np.asarray(theta, dtype=float)
    step, w = np.deg2rad(step_deg), np.deg2rad(width_deg)
    with np.errstate(invalid="ignore"):
        dth = np.abs(theta - np.round(theta / step) * step)
        pr = np.asarray(phi, dtype=float) / step
        dph = np.abs(pr - np.round(pr)) * step
        return (dth <= w) | (dph * np.sin(theta) <= w)


def light_weight(
    theta: np.ndarray, phi: np.ndarray, theta_behind: float, radius_deg: float, alpha: float
) -> np.ndarray:
    """Glow weight `alpha * exp(-d^2 / (2 sigma^2))` around the behind-observer point.

    `d` is the great-circle angle to (`theta_behind`, phi = pi); NaN angles give weight 0.
    """
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    sigma = np.deg2rad(radius_deg)
    with np.errstate(invalid="ignore"):
        cosd = np.cos(theta) * np.cos(theta_behind) + np.sin(theta) * np.sin(theta_behind) * np.cos(
            phi - np.pi
        )
        d = np.arccos(np.clip(cosd, -1.0, 1.0))
        w = alpha * np.exp(-(d**2) / (2 * sigma**2))
    return np.nan_to_num(w, nan=0.0)


def sky_directions(out: dict[str, np.ndarray], cfg: ShadowConfig) -> tuple[np.ndarray, np.ndarray]:
    """Escape angles (theta, phi) per pixel from a run's arrays; NaN where the ray did not escape.

    Kerr: theta = coord2, phi = phi. Weyl: theta = pi - atan2(rho, z) from coord1/coord2, and phi is
    a flat-sky approximation `pi + PHI_LEFT_SIGN * beta` of the pixel column (no phi is traced).
    """
    esc = np.asarray(out["status"]) == 0
    if "phi" in out:
        theta, phi = np.asarray(out["coord2"], float), np.asarray(out["phi"], float)
    else:
        # _mirror4 negates z on the lower array rows (drawn at the image top), so flip theta
        theta = np.pi - np.arctan2(out["coord1"], out["coord2"])
        beta = np.linspace(cfg.half_angle, -cfg.half_angle, esc.shape[1])
        phi = np.broadcast_to(np.pi + PHI_LEFT_SIGN * beta, esc.shape)
    return np.where(esc, theta, np.nan), np.where(esc, np.mod(phi, _TWO_PI), np.nan)


def compose_rgb(
    status: np.ndarray, theta: np.ndarray, phi: np.ndarray, spec: VisualSpec
) -> np.ndarray:
    """Compose the (n, n, 3) float RGB image in [0, 1] for `spec`.

    Captured pixels are black; failed pixels (status 3) are red. Disk pixels are opaque white for
    greyscale, black for black_hole, else `alpha*white + (1-alpha)*DISK_BACKDROP`.
    """
    status = np.asarray(status)
    escaped, disk, failed = status == 0, status == 2, status == 3

    if spec.background == "black_hole":
        rgb = np.where(escaped[..., None], WHITE, BLACK)
        return np.where(failed[..., None], RED, rgb)

    sky = sky_rgb(theta, phi, spec)
    if spec.light_source:
        w = light_weight(
            theta, phi, np.pi - spec.theta_obs, spec.light_radius_deg, spec.light_alpha
        )[..., None]
        sky = w * WHITE + (1 - w) * sky
    if spec.grid:
        sky = np.where(
            grid_mask(theta, phi, spec.step_deg, spec.width_deg)[..., None], GRID_COLOR, sky
        )
    if spec.background == "greyscale":
        disk_rgb = WHITE
    else:
        disk_rgb = spec.disk_alpha * WHITE + (1 - spec.disk_alpha) * DISK_BACKDROP

    rgb = np.broadcast_to(BLACK, sky.shape).copy()
    rgb = np.where(escaped[..., None], sky, rgb)
    rgb = np.where(disk[..., None], disk_rgb, rgb)
    rgb = np.where(failed[..., None], RED, rgb)
    return np.clip(rgb, 0.0, 1.0)


def render_image(
    out: dict[str, np.ndarray], cfg: ShadowConfig, path: str | Path | None = None
) -> None:
    """Render a run's arrays with `cfg`'s visual options; save to `path`, show if `cfg.show`."""
    import matplotlib.pyplot as plt

    spec = VisualSpec.from_config(cfg)
    status = out["status"]
    theta, phi = sky_directions(out, cfg)
    rgb = compose_rgb(status, theta, phi, spec)
    h = cfg.half_angle
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(rgb, origin="lower", extent=(-h, h, -h, h))
    plt.title(f"$M={cfg.M}$, $a={cfg.a}$, $M_D={cfg.MD}$, $b={cfg.b}$ ({cfg.backend})", size=20)
    plt.xlabel("$-\\beta$", size=20)
    plt.ylabel("$\\alpha$", size=20)
    if path is not None:
        fig.savefig(path, dpi=150)
    if cfg.show:
        plt.show()
    plt.close(fig)
