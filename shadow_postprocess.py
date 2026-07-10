"""Shared post-processing helpers for quarter-plane shadow ray-tracing outputs.

The sibling scripts (simetria_shadow.py, simetria_shadow_v2.py, symmetry.py) each
ray-trace only one quarter of the image plane (by symmetry of the underlying
spacetime) and then (a) mirror that quarter into the full 2N x 2M plane and
(b) classify each pixel as "captured by the black hole", "beyond the disk", or
neither. This module factors those two operations out into small, vectorized,
side-effect-free numpy functions, preserving the exact (occasionally
asymmetric) behavior of the original per-script loop implementations.
"""

import numpy as np


def mirror_quadrants(A, antisymmetric=False):
    """Mirror a ray-traced quarter-plane array into the full 2N x 2M plane.

    Reflects A across both axes to reconstruct the full symmetric image:
    top-left = A, top-right = A mirrored left-right, bottom-left = A mirrored
    top-bottom, bottom-right = A mirrored both ways. If antisymmetric is True,
    the bottom half (mirrored top-bottom, i.e. z -> -z) is negated -- used for
    quantities like Mz that are antisymmetric under that reflection.

    Args:
        A: 2D numpy array of shape (n, m) holding one quarter-plane of the
            ray-traced image.
        antisymmetric: If True, negate the bottom half (rows mirrored
            top-to-bottom) to reflect a quantity that flips sign under
            z -> -z, rather than one that is invariant under it.

    Returns:
        A new 2D numpy array of shape (2 * n, 2 * m) holding the full mirrored
        plane.
    """
    n, m = A.shape
    out = np.zeros((2 * n, 2 * m))
    sign = -1 if antisymmetric else 1
    out[:n, :m] = A
    out[:n, m:] = A[:, ::-1]
    out[n:, :m] = sign * A[::-1, :]
    out[n:, m:] = sign * A[::-1, ::-1]
    return out


def classify_shadow(mat, mz, b, capture_r=0.002, mz_escape=49.0,
                     unshift_mat=False, use_abs_z=True):
    """Classify each pixel of a shadow image as captured/beyond-disk/neither.

    Returns an array of the same shape as mat with values:
      1 = captured by the black hole (mat <= capture_r, after optional unshift)
      2 = beyond the disk (mat > b AND z escapes past mz_escape)
      0 = neither
    Capture (1) takes priority over beyond-disk (2) when both would match,
    matching the original if/elif ordering in the source scripts.

    If unshift_mat is True, values of mat greater than 50 are first reduced
    by 50 (undoing a wraparound offset used by one ray tracer's output
    convention) before the capture/beyond-disk comparisons.
    If use_abs_z is True, the escape test uses abs(mz) > mz_escape (i.e. checks
    both z > mz_escape and z < -mz_escape); if False, it uses the raw mz > mz_escape
    only (no lower-bound check) -- this matches a real difference between the
    two script families' original logic, so do not "fix" it to be symmetric.

    Args:
        mat: 2D numpy array of radial/impact-parameter values per pixel.
        mz: 2D numpy array (same shape as mat) of the z-coordinate (or
            related escape quantity) per pixel.
        b: Scalar threshold; mat > b is required (together with the z escape
            test) for a pixel to be classified as beyond the disk.
        capture_r: Scalar threshold; mat <= capture_r marks a pixel as
            captured by the black hole.
        mz_escape: Scalar threshold on mz (or abs(mz)) for the beyond-disk
            escape test.
        unshift_mat: If True, subtract 50 from entries of mat greater than 50
            before applying the thresholds.
        use_abs_z: If True, compare abs(mz) against mz_escape; if False,
            compare the raw mz against mz_escape (no lower bound).

    Returns:
        A new numpy array of the same shape as mat with values in {0, 1, 2}.
    """
    mat_u = np.where(mat > 50, mat - 50, mat) if unshift_mat else mat
    z = np.abs(mz) if use_abs_z else mz
    M = np.zeros_like(mat_u)
    M[mat_u <= capture_r] = 1
    M[(M != 1) & (mat_u > b) & (z > mz_escape)] = 2
    return M
