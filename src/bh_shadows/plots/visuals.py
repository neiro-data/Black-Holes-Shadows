"""Selectable image visuals: colour each pixel from its ray outcome and escape direction.

Post-process only: it reads the arrays the tracers already return (no extra tracing). Pure numpy
except `render_image` and `load_sky_image` (matplotlib). `status` is 0 escaped, 1 captured,
2 disk, 3 failed. Escape angles `theta` in [0, pi], `phi` in [0, 2*pi) come from `sky_directions`:
Kerr uses `coord2` and `phi`; Weyl uses theta = pi - atan2(rho, z) (rows are z-mirrored) and the
traced `phi`. A legacy Weyl npz without `phi` falls back to a flat-sky phi from the pixel column
(beta).
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

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bh_shadows.runner.shadow_config import ShadowConfig

log = logging.getLogger(__name__)

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

# `grid_mode = "end_z"` (Weyl): z lines every Z_LINE_SPACING end-z units from Z_LINE_TOP, half
# width END_Z_DZ; phi lines have half width END_Z_DANG (rad).
Z_LINE_TOP = 30.0
Z_LINE_SPACING = 60.0 / 18.0
END_Z_DZ = 0.06
END_Z_DANG = 0.01

# `grid_mode = "flat_angle"` (Weyl): lines on the flat-space angles (alpha_e, beta_e) of the ray end
# point, same half width (rad) in both directions: 1.5 * (half the central end-z phi line width).
FLAT_ANGLE_HALF_WIDTH = 1.5 * 0.0123 / 4


@dataclass(frozen=True)
class VisualSpec:
    """Visual options for `compose_rgb`. The fields follow the `[plot]` config keys."""

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
    grid_mode: str = "theta"
    distance: float = 15.0

    @classmethod
    def from_config(cls, cfg) -> VisualSpec:
        """Make a spec from a config.

        Parameters
        ----------
        cfg : ShadowConfig
            Run config. The function loads the sky image when `background == "image"`.

        Returns
        -------
        VisualSpec
            Visual options.
        """
        sky = load_sky_image(cfg.image_path) if cfg.background == "image" else None
        return cls(
            background=cfg.background,
            grid=cfg.grid,
            step_deg=cfg.grid_step_deg,
            width_deg=cfg.grid_width_deg,
            grid_mode=cfg.grid_mode,
            distance=cfg.distance,
            disk_alpha=cfg.disk_alpha,
            sky=sky,
            light_source=cfg.light_source,
            light_alpha=cfg.light_alpha,
            light_radius_deg=cfg.light_radius_deg,
            theta_obs=np.pi / 2 if cfg.backend == "weyl" else float(np.deg2rad(cfg.theta0)),
        )


def load_sky_image(path: str | Path) -> np.ndarray:
    """Read a sky image.

    Parameters
    ----------
    path : str or Path
        PNG or JPEG file.

    Returns
    -------
    np.ndarray
        Float RGB array of shape (H, W, 3) in [0, 1]. Greyscale becomes RGB. Alpha is removed.
    """
    import matplotlib.pyplot as plt

    img = np.asarray(plt.imread(path))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3]
    if img.dtype == np.uint8:
        return img.astype(float) / 255.0
    return np.clip(img.astype(float), 0.0, 1.0)


def _phi_rel(phi: np.ndarray) -> np.ndarray:
    """Wrap `phi - pi` into (-pi, pi].

    Parameters
    ----------
    phi : np.ndarray
        Azimuth in radians.

    Returns
    -------
    np.ndarray
        Azimuth relative to the direction behind the observer.
    """
    return np.pi - np.mod(np.pi - (np.asarray(phi, dtype=float) - np.pi), _TWO_PI)


def sky_rgb(theta: np.ndarray, phi: np.ndarray, spec: VisualSpec) -> np.ndarray:
    """Return the sky colour for each escape direction.

    Parameters
    ----------
    theta, phi : np.ndarray
        Escape angles in radians. NaN angles give grey (or pixel (0, 0) of the image).
    spec : VisualSpec
        Visual options. Uses `background` and `sky`.

    Returns
    -------
    np.ndarray
        RGB values of shape `theta.shape + (3,)`.

    Raises
    ------
    ValueError
        If the background has no sky, or if `background == "image"` and `spec.sky` is None.
    """
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
    """Return the mask of the celestial-sphere grid lines.

    A theta line is on `|theta - k*step| <= w`. A phi line is on `|dphi| * sin(theta) <= w`, which
    keeps the line width constant near the poles.

    Parameters
    ----------
    theta, phi : np.ndarray
        Escape angles in radians.
    step_deg : float
        Line spacing in degrees.
    width_deg : float
        Half width of a line in degrees.

    Returns
    -------
    np.ndarray
        True on a grid line.
    """
    theta = np.asarray(theta, dtype=float)
    step, w = np.deg2rad(step_deg), np.deg2rad(width_deg)
    with np.errstate(invalid="ignore"):
        dth = np.abs(theta - np.round(theta / step) * step)
        pr = np.asarray(phi, dtype=float) / step
        dph = np.abs(pr - np.round(pr)) * step
        return (dth <= w) | (dph * np.sin(theta) <= w)


def grid_mask_end_z(z_end: np.ndarray, phi: np.ndarray, step_deg: float) -> np.ndarray:
    """Return the mask of the Weyl end-z grid lines.

    A z line is on `|z_end - (Z_LINE_TOP - k*Z_LINE_SPACING)| <= END_Z_DZ` for k = 0..18. A phi
    line is on `|phi - k*step| <= END_Z_DANG`.

    Parameters
    ----------
    z_end : np.ndarray
        End z of each ray (`coord2`).
    phi : np.ndarray
        Escape azimuth in radians.
    step_deg : float
        Phi line spacing in degrees.

    Returns
    -------
    np.ndarray
        True on a grid line. NaN inputs give False.
    """
    z_end = np.asarray(z_end, dtype=float)
    step = np.deg2rad(step_deg)
    with np.errstate(invalid="ignore"):
        k = np.clip(np.round((Z_LINE_TOP - z_end) / Z_LINE_SPACING), 0, 18)
        dz = np.abs(z_end - (Z_LINE_TOP - k * Z_LINE_SPACING))
        pr = np.asarray(phi, dtype=float) / step
        dph = np.abs(pr - np.round(pr)) * step
        return (dz <= END_Z_DZ) | (dph <= END_Z_DANG)


def flat_end_angles(
    rho_end: np.ndarray, z_end: np.ndarray, phi: np.ndarray, distance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the flat-space pixel angles that would reach each ray end point.

    The end point `(rho, phi, -z)` is joined to the observer at `x = distance` by a straight line.
    The angles of that line (sign convention of the traced runs) are `alpha_e`, `beta_e`. Without
    a black hole they equal the pixel angles, so lines of constant `alpha_e` or `beta_e` are
    straight; the black hole bends them.

    Parameters
    ----------
    rho_end, z_end, phi : np.ndarray
        End rho, end z and end phi of each ray (`coord1`, `coord2`, `phi`).
    distance : float
        Observer distance.

    Returns
    -------
    alpha_e, beta_e : np.ndarray
        Angles in radians. NaN inputs give NaN.
    """
    rho_end = np.asarray(rho_end, dtype=float)
    dx = rho_end * np.cos(phi) - distance
    dy = rho_end * np.sin(phi)
    dz = -np.asarray(z_end, dtype=float)
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.arcsin(dz / norm), np.arctan2(-dy, -dx)


def grid_mask_flat_angle(
    rho_end: np.ndarray, z_end: np.ndarray, phi: np.ndarray, distance: float, step_deg: float
) -> np.ndarray:
    """Return the mask of the lensed flat-angle grid lines.

    Lines are at `alpha_e = k*step` and `beta_e = k*step` (see `flat_end_angles`), both with half
    width `FLAT_ANGLE_HALF_WIDTH`. They are straight without a black hole and bend with it.

    Parameters
    ----------
    rho_end, z_end, phi : np.ndarray
        End rho, end z and end phi of each ray.
    distance : float
        Observer distance.
    step_deg : float
        Line spacing in degrees.

    Returns
    -------
    np.ndarray
        True on a grid line. NaN inputs give False.
    """
    step = np.deg2rad(step_deg)
    ae, be = flat_end_angles(rho_end, z_end, phi, distance)
    with np.errstate(invalid="ignore"):
        on = lambda x: np.abs(x - np.round(x / step) * step) <= FLAT_ANGLE_HALF_WIDTH  # noqa: E731
        return on(ae) | on(be)


def light_weight(
    theta: np.ndarray, phi: np.ndarray, theta_behind: float, radius_deg: float, alpha: float
) -> np.ndarray:
    """Return the weight of the light-source glow.

    The weight is `alpha * exp(-d**2 / (2 * sigma**2))`. `d` is the great-circle angle to the point
    (`theta_behind`, phi = pi).

    Parameters
    ----------
    theta, phi : np.ndarray
        Escape angles in radians.
    theta_behind : float
        Polar angle of the glow centre, in radians.
    radius_deg : float
        Glow radius `sigma` in degrees.
    alpha : float
        Peak weight.

    Returns
    -------
    np.ndarray
        Weight in [0, alpha]. NaN angles give 0.
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
    """Return the escape angles of each pixel.

    Kerr uses `theta = coord2` and the traced `phi`. Weyl uses `theta = pi - atan2(rho, z)` and the
    traced `phi`. A legacy Weyl dict without `phi` uses the flat-sky value
    `pi + PHI_LEFT_SIGN * beta` and the function logs a warning.

    Parameters
    ----------
    out : dict of str to np.ndarray
        Run arrays (`status`, `coord1`, `coord2`, and `phi`).
    cfg : ShadowConfig
        Run config. Uses `half_angle`.

    Returns
    -------
    theta, phi : np.ndarray
        Escape angles. NaN where the ray did not escape.
    """
    esc = np.asarray(out["status"]) == 0
    if cfg.backend == "kerr":
        theta, phi = np.asarray(out["coord2"], float), np.asarray(out["phi"], float)
    else:
        # _mirror4 negates z on the lower array rows (drawn at the image top), so flip theta
        theta = np.pi - np.arctan2(out["coord1"], out["coord2"])
        if "phi" in out:
            phi = np.asarray(out["phi"], float)
            return np.where(esc, theta, np.nan), np.where(esc, np.mod(phi, _TWO_PI), np.nan)
        log.warning("Weyl output has no traced phi (legacy npz): using flat-sky phi")
        beta = np.linspace(cfg.half_angle, -cfg.half_angle, esc.shape[1])
        phi = np.broadcast_to(np.pi + PHI_LEFT_SIGN * beta, esc.shape)
    return np.where(esc, theta, np.nan), np.where(esc, np.mod(phi, _TWO_PI), np.nan)


def compose_rgb(
    status: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    spec: VisualSpec,
    z_end: np.ndarray | None = None,
    rho_end: np.ndarray | None = None,
) -> np.ndarray:
    """Compose the RGB image.

    Order of layers: sky, glow, grid, then captured, disk and failed pixels. Captured pixels are
    black. Failed pixels (status 3) are red. Disk pixels are white for "greyscale", black for
    "black_hole", else `disk_alpha * white + (1 - disk_alpha) * DISK_BACKDROP`.

    Parameters
    ----------
    status : np.ndarray
        Status of each pixel: 0 escaped, 1 captured, 2 disk, 3 failed.
    theta, phi : np.ndarray
        Escape angles from `sky_directions`.
    spec : VisualSpec
        Visual options.
    z_end : np.ndarray, optional
        End z of each ray. Needed when `spec.grid_mode` is `end_z` or `flat_angle`.
    rho_end : np.ndarray, optional
        End rho of each ray. Needed when `spec.grid_mode == "flat_angle"`.

    Returns
    -------
    np.ndarray
        Float RGB image of shape (n, n, 3) in [0, 1].
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
        if spec.grid_mode == "flat_angle":
            if z_end is None or rho_end is None:
                raise ValueError("grid_mode 'flat_angle' needs rho_end and z_end")
            mask = grid_mask_flat_angle(rho_end, z_end, phi, spec.distance, spec.step_deg)
        elif spec.grid_mode == "end_z":
            if z_end is None:
                raise ValueError("grid_mode 'end_z' needs z_end")
            mask = grid_mask_end_z(z_end, phi, spec.step_deg)
        else:
            mask = grid_mask(theta, phi, spec.step_deg, spec.width_deg)
        sky = np.where(mask[..., None], GRID_COLOR, sky)
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
    """Render the run arrays with the visual options of the config.

    Parameters
    ----------
    out : dict of str to np.ndarray
        Run arrays.
    cfg : ShadowConfig
        Run config.
    path : str or Path, optional
        Save the figure to this path. The figure shows on screen if `cfg.show` is set.
    """
    import matplotlib.pyplot as plt

    spec = VisualSpec.from_config(cfg)
    status = out["status"]
    theta, phi = sky_directions(out, cfg)
    rgb = compose_rgb(status, theta, phi, spec, z_end=out.get("coord2"), rho_end=out.get("coord1"))
    h = cfg.half_angle
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(rgb, origin="lower", extent=(-h, h, -h, h))
    if cfg.shadow_edge:
        from bh_shadows.core.shadow_edge import draw_edge

        draw_edge(plt.gca(), cfg)
        plt.xlim(-h, h)
        plt.ylim(-h, h)
    plt.title(f"$M={cfg.M}$, $a={cfg.a}$, $M_D={cfg.MD}$, $b={cfg.b}$ ({cfg.backend})", size=20)
    plt.xlabel("$-\\beta$", size=20)
    plt.ylabel("$\\alpha$", size=20)
    if path is not None:
        fig.savefig(path, dpi=150)
    if cfg.show:
        plt.show()
    plt.close(fig)
