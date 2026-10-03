"""Shared plotting / colormap / classification helpers for the Weyl-family driver scripts.

Consolidates the colormap palettes and the Mat/Mz -> {shadow, disk, quadrant, grid-lines}
classification logic that used to be copy-pasted (mostly inside commented-out blocks)
across `test_parallel_SHADOW.py`, `test_Z_SHADOW.py`, `symmetry.py`, `simetria_shadow.py`,
`simetria_shadow_v2.py`, and `lensing_image.py`.

Every colormap option that existed anywhere in the legacy scripts is kept here as a
named palette so no visual option is lost.
"""

from __future__ import annotations

import matplotlib.colors
import numpy as np

# ---------------------------------------------------------------------------
# Palettes (all colormap options that existed anywhere in the legacy scripts)
# ---------------------------------------------------------------------------
PALETTES: dict[str, list[list[float]]] = {
    # Plain black & white: [background/shadow, BH+grid]
    "black_white": [[1, 1, 1], [0, 0, 0]],
    # White background, black BH, dark-grey secondary class (e.g. disk or Einstein ring)
    "white_black_grey": [[1, 1, 1], [0, 0, 0], [0.192, 0.192, 0.192]],
    # Grey background, black BH/grid, white secondary class
    "grey_black_white": [[0.505, 0.505, 0.505], [0, 0, 0], [1, 1, 1]],
    # Dark-grey background, black BH/grid (used by symmetry.py's default)
    "darkgrey_black": [[0.192, 0.192, 0.192], [0, 0, 0]],
    # Per-quadrant lensed-disk colouring: black = BH + grid lines, colours = the four
    # image quadrants (red, green, blue, yellow)
    "quadrant": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]],
}


def get_colormap(name: str) -> matplotlib.colors.ListedColormap:
    """Build a matplotlib ListedColormap from one of the named PALETTES."""
    if name not in PALETTES:
        raise KeyError(f"Unknown palette {name!r}. Available: {sorted(PALETTES)}")
    return matplotlib.colors.ListedColormap(PALETTES[name])


# ---------------------------------------------------------------------------
# Mat/Mz -> classification helpers
# ---------------------------------------------------------------------------
def classify_shadow_disk(
    Mat: np.ndarray, Mz: np.ndarray, b: float, rho_tol: float = 0.002
) -> np.ndarray:
    """Classify each grid point as background (0), BH shadow (1), or disk hit (2).

    Mirrors the `M2` classification loop used by test_parallel_SHADOW.py / test_Z_SHADOW.py /
    symmetry.py: BH capture is flagged by `func`'s [0.001, 0.001] sentinel (rho <= rho_tol
    after un-offsetting), and disk crossings are flagged by the [rho, z+50] sentinel used
    by `weyl_core.func_bh_disk_crossing` (rho > b and z > 49 after the +50 offset).
    """
    Mat2 = np.where(Mat > 50.0, Mat - 50.0, Mat)

    M2 = np.zeros_like(Mat)
    M2[Mat2 <= rho_tol] = 1
    M2[(Mat2 > b) & (Mz > 49.0)] = 2
    return M2


STATUS_ESCAPED, STATUS_CAPTURED, STATUS_DISK, STATUS_FAILED = 0, 1, 2, 3


def plot_status(status: np.ndarray, cfg, path=None):
    """Plot a full-image status grid (0 escaped, 1 captured, 2 disk, 3 NaN failure).

    Rows run over alpha (bottom to top), columns over -beta. Failed pixels are drawn red.
    Saves to `path` when given, shows the figure when `cfg.show` is set.
    """
    import matplotlib.pyplot as plt

    h = cfg.half_angle
    cm = get_colormap(cfg.palette).copy()
    cm.set_bad("red")
    img = np.ma.masked_where(status == STATUS_FAILED, status)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(
        img,
        cmap=cm,
        vmin=0,
        vmax=len(PALETTES[cfg.palette]) - 1,
        origin="lower",
        extent=[-h, h, -h, h],
    )
    plt.title(f"$M={cfg.M}$, $a={cfg.a}$, $M_D={cfg.MD}$, $b={cfg.b}$ ({cfg.backend})", size=20)
    plt.xlabel("$-\\beta$", size=20)
    plt.ylabel("$\\alpha$", size=20)
    if path is not None:
        fig.savefig(path, dpi=150)
    if cfg.show:
        plt.show()
    plt.close(fig)


def mirror_quadrants(M2: np.ndarray) -> np.ndarray:
    """Mirror a single-quadrant classification grid into the full 4-quadrant image.

    Equivalent to the `Msym`/`Mat2` mirroring loops in symmetry.py, simetria_shadow.py,
    and simetria_shadow_v2.py.
    """
    n_alfa, n_beta = M2.shape
    Msym = np.zeros((2 * n_alfa, 2 * n_beta))
    Msym[:n_alfa, :n_beta] = M2
    Msym[:n_alfa, n_beta:] = M2[:, ::-1]
    Msym[n_alfa:, :n_beta] = M2[::-1, :]
    Msym[n_alfa:, n_beta:] = M2[::-1, ::-1]
    return Msym


def mirror_raw_and_classify(
    Mat: np.ndarray, Mz: np.ndarray, b: float, rho_tol: float = 0.002
) -> np.ndarray:
    """Mirror the raw Mat/Mz matrices to 4 quadrants (negating Mz for the flipped half),
    then classify. Used by simetria_shadow.py / simetria_shadow_v2.py, which mirror
    before classifying (as opposed to `classify_shadow_disk` + `mirror_quadrants`, which
    classify first and mirror the class labels).
    """
    n_alfa, n_beta = Mat.shape
    Mat2 = np.zeros((2 * n_alfa, 2 * n_beta))
    Mz2 = np.zeros((2 * n_alfa, 2 * n_beta))

    Mat2[:n_alfa, :n_beta] = Mat
    Mat2[:n_alfa, n_beta:] = Mat[:, ::-1]
    Mat2[n_alfa:, :n_beta] = Mat[::-1, :]
    Mat2[n_alfa:, n_beta:] = Mat[::-1, ::-1]

    Mz2[:n_alfa, :n_beta] = Mz
    Mz2[:n_alfa, n_beta:] = Mz[:, ::-1]
    Mz2[n_alfa:, :n_beta] = -Mz[::-1, :]
    Mz2[n_alfa:, n_beta:] = -Mz[::-1, ::-1]

    M2 = np.zeros_like(Mat2)
    M2[Mat2 <= rho_tol] = 1
    M2[(Mat2 > b) & ((Mz2 > 49.0) | (Mz2 < -49.0))] = 2
    return M2


def normalize_phi(Mphi: np.ndarray) -> np.ndarray:
    """Wrap an accumulated phi coordinate matrix into [0, 2*pi)."""
    two_pi = 2 * np.pi
    return np.mod(Mphi, two_pi)


def classify_lensing_quadrants(
    Mat: np.ndarray, Mz: np.ndarray, Mphi: np.ndarray, rho_tol: float = 0.002
) -> np.ndarray:
    """Classify each grid point into background (0) or one of the 4 disk-image quadrants (1-4).

    Mirrors the classification loop in test_symmetry_lensing.py / lensing_image.py that
    combines the sign of z (upper/lower disk face) with the wrapped phi half (0..pi vs
    pi..2*pi) to number the four lensed-image quadrants.
    """
    Mphi2 = normalize_phi(Mphi)
    M2 = np.zeros_like(Mat)

    below = Mz < 0
    above = ~below
    phi_lo = (Mphi2 >= 0) & (Mphi2 <= np.pi)
    phi_hi = (Mphi2 > np.pi) & (Mphi2 <= 2 * np.pi)

    M2[below & phi_lo] = 3
    M2[below & phi_hi] = 4
    M2[above & phi_lo] = 2
    M2[above & phi_hi] = 1
    M2[Mat < rho_tol] = 0
    return M2


def add_grid_lines(
    M2: np.ndarray,
    the_or_z: np.ndarray,
    Mphi2: np.ndarray,
    kmax_line1: float,
    kmax_phi: float,
    step_line1_deg: float = 10.0,
    step_phi_deg: float = 10.0,
    dang: float = 0.01,
) -> np.ndarray:
    """Overlay black grid lines (class 0) at fixed theta/z and phi intervals.

    Mirrors the "Black lines of the grid" loops present (commented out) in several
    legacy plotting scripts. `the_or_z` is either a theta matrix (radians) or a z matrix
    depending on which coordinate the caller wants grid lines drawn in.
    """
    out = M2.copy()
    step1 = np.deg2rad(step_line1_deg)
    step_phi = np.deg2rad(step_phi_deg)

    for k in range(int(kmax_line1) + 1):
        target = k * step1
        out[np.abs(the_or_z - target) <= dang] = 0

    for k in range(int(kmax_phi) + 1):
        target = k * step_phi
        out[np.abs(Mphi2 - target) <= dang] = 0

    return out


def load_grid_from_image(image_path: str, threshold: float = 0.5) -> np.ndarray:
    """Optional hook: build a boolean grid mask from a user-supplied image.

    No caller in this codebase currently builds a grid this way -- kept minimal, as a
    documented extension point. Requires matplotlib's image reader; pixels brighter than
    `threshold` (after converting to greyscale) map to True.
    """
    import matplotlib.pyplot as plt

    img = plt.imread(image_path)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1)
    return img > threshold
