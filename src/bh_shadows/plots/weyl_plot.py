"""Palettes, status plots and classification helpers for the Weyl tracers.

`PALETTES` keeps every colormap of the original scripts.
"""

from __future__ import annotations

import matplotlib.colors
import numpy as np

# ---------------------------------------------------------------------------
# Palettes (all colormap options of the original scripts)
# ---------------------------------------------------------------------------
PALETTES: dict[str, list[list[float]]] = {
    # Plain black & white: [background/shadow, BH+grid]
    "black_white": [[1, 1, 1], [0, 0, 0]],
    # White background, black BH, dark-grey secondary class (e.g. disk or Einstein ring)
    "white_black_grey": [[1, 1, 1], [0, 0, 0], [0.192, 0.192, 0.192]],
    # Grey background, black BH/grid, white secondary class
    "grey_black_white": [[0.505, 0.505, 0.505], [0, 0, 0], [1, 1, 1]],
    # Dark-grey background, black BH/grid (default of the original plot script)
    "darkgrey_black": [[0.192, 0.192, 0.192], [0, 0, 0]],
    # Per-quadrant lensed-disk colouring: black = BH + grid lines, colours = the four
    # image quadrants (red, green, blue, yellow)
    "quadrant": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]],
}


def get_colormap(name: str) -> matplotlib.colors.ListedColormap:
    """Return a colormap from `PALETTES`.

    Parameters
    ----------
    name : str
        Palette name.

    Returns
    -------
    matplotlib.colors.ListedColormap
        Colormap with one colour for each class.

    Raises
    ------
    KeyError
        If `name` is not in `PALETTES`.
    """
    if name not in PALETTES:
        raise KeyError(f"Unknown palette {name!r}. Available: {sorted(PALETTES)}")
    return matplotlib.colors.ListedColormap(PALETTES[name])


# ---------------------------------------------------------------------------
# Mat/Mz -> classification helpers
# ---------------------------------------------------------------------------
def classify_shadow_disk(
    Mat: np.ndarray, Mz: np.ndarray, b: float, rho_tol: float = 0.002
) -> np.ndarray:
    """Classify each Weyl pixel from its end state.

    The function reads the sentinels of `weyl_core.func`: capture gives `[0.001, 0.001]` and a disk
    hit gives `[rho, z + 50]`.

    Parameters
    ----------
    Mat, Mz : np.ndarray
        End rho and end z of each ray.
    b : float
        Disk inner radius.
    rho_tol : float, optional
        Largest end rho that counts as capture.

    Returns
    -------
    np.ndarray
        0 background, 1 shadow, 2 disk.
    """
    Mat2 = np.where(Mat > 50.0, Mat - 50.0, Mat)

    M2 = np.zeros_like(Mat)
    M2[Mat2 <= rho_tol] = 1
    M2[(Mat2 > b) & (Mz > 49.0)] = 2
    return M2


STATUS_ESCAPED, STATUS_CAPTURED, STATUS_DISK, STATUS_FAILED = 0, 1, 2, 3


def plot_status(status: np.ndarray, cfg, path=None):
    """Plot a status image with the config palette.

    Rows go over alpha (bottom to top). Columns go over -beta. Failed pixels (status 3) are red.

    Parameters
    ----------
    status : np.ndarray
        Full image: 0 escaped, 1 captured, 2 disk, 3 failed.
    cfg : ShadowConfig
        Run config. Uses `palette`, `half_angle`, `show`, `shadow_edge` and the title values.
    path : str or Path, optional
        Save the figure to this path.
    """
    import matplotlib.pyplot as plt

    h = cfg.half_angle
    cm = get_colormap(cfg.palette).with_extremes(bad="red")
    img = np.ma.masked_where(status == STATUS_FAILED, status)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(
        img,
        cmap=cm,
        vmin=0,
        vmax=len(PALETTES[cfg.palette]) - 1,
        origin="lower",
        extent=(-h, h, -h, h),
    )
    if getattr(cfg, "shadow_edge", False):
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


def mirror_quadrants(M2: np.ndarray) -> np.ndarray:
    """Mirror one quadrant into the full image.

    Parameters
    ----------
    M2 : np.ndarray
        Quadrant of shape (na, nb).

    Returns
    -------
    np.ndarray
        Full image of shape (2 * na, 2 * nb).
    """
    n_alpha, n_beta = M2.shape
    Msym = np.zeros((2 * n_alpha, 2 * n_beta))
    Msym[:n_alpha, :n_beta] = M2
    Msym[:n_alpha, n_beta:] = M2[:, ::-1]
    Msym[n_alpha:, :n_beta] = M2[::-1, :]
    Msym[n_alpha:, n_beta:] = M2[::-1, ::-1]
    return Msym
