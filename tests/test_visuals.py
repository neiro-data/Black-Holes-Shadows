"""Pure visuals tests on synthetic arrays (no tracing)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import visuals as v

N = 4
HALF_PI = np.pi / 2


def spec(**kw):
    return v.VisualSpec(**kw)


def full(value, n=N):
    return np.full((n, n), value, dtype=float)


def test_black_hole_only_black_and_white():
    status = np.array([[0, 1], [2, 0]])
    rgb = v.compose_rgb(status, full(1.0, 2), full(1.0, 2), spec(background="black_hole"))
    assert rgb.shape == (2, 2, 3)
    assert np.array_equal(rgb[0, 0], v.WHITE) and np.array_equal(rgb[1, 1], v.WHITE)
    assert np.array_equal(rgb[0, 1], v.BLACK) and np.array_equal(rgb[1, 0], v.BLACK)


def test_two_color_upper_red_lower_blue():
    theta = np.array([[0.5, 0.5], [2.5, 2.5]])
    rgb = v.compose_rgb(np.zeros((2, 2), int), theta, full(1.0, 2), spec(background="two_color"))
    assert np.array_equal(rgb[0, 0], v.RED) and np.array_equal(rgb[0, 1], v.RED)
    assert np.array_equal(rgb[1, 0], v.BLUE) and np.array_equal(rgb[1, 1], v.BLUE)


def test_four_color_quadrants():
    # phi_rel = phi - pi: PHI_LEFT_SIGN * rel > 0 is "left"
    left, right = np.pi + 0.5 * v.PHI_LEFT_SIGN, np.pi - 0.5 * v.PHI_LEFT_SIGN
    theta = np.array([[0.5, 0.5], [2.5, 2.5]])
    phi = np.array([[left, right], [left, right]])
    rgb = v.compose_rgb(np.zeros((2, 2), int), theta, phi, spec(background="four_color"))
    assert np.array_equal(rgb[0, 0], v.RED) and np.array_equal(rgb[0, 1], v.GREEN)
    assert np.array_equal(rgb[1, 0], v.YELLOW) and np.array_equal(rgb[1, 1], v.BLUE)


def test_greyscale_sky_grey_and_disk_pure_white():
    status = np.array([[0, 2], [1, 0]])
    rgb = v.compose_rgb(status, full(1.0, 2), full(1.0, 2), spec(background="greyscale"))
    assert np.array_equal(rgb[0, 1], v.WHITE)
    assert np.array_equal(rgb[0, 0], v.GREY)
    assert np.array_equal(rgb[1, 0], v.BLACK)


@pytest.mark.parametrize("alpha", [0.0, 0.3, 1.0])
def test_disk_alpha_blend_exact(alpha):
    status = np.full((2, 2), 2)
    rgb = v.compose_rgb(
        status, full(1.0, 2), full(1.0, 2), spec(background="two_color", disk_alpha=alpha)
    )
    expected = alpha * v.WHITE + (1 - alpha) * v.DISK_BACKDROP
    assert np.allclose(rgb, expected)


def test_status3_is_red_in_every_background():
    status = np.full((2, 2), 3)
    for bg in ("black_hole", "two_color", "greyscale", "four_color"):
        rgb = v.compose_rgb(status, full(np.nan, 2), full(np.nan, 2), spec(background=bg))
        assert np.array_equal(rgb[0, 0], v.RED), bg


def test_nan_angles_on_escaped_pixels_do_not_crash():
    rgb = v.compose_rgb(np.zeros((2, 2), int), full(np.nan, 2), full(np.nan, 2), spec())
    assert np.isfinite(rgb).all()


def test_grid_mask_hits_multiples_and_misses_mid_cells():
    step = 10.0
    on = np.deg2rad([30.0, 60.0, 90.0])  # theta at multiples
    mid = np.deg2rad([35.0, 55.0, 85.0])
    phi_mid = np.deg2rad(35.0)
    assert v.grid_mask(on, np.full(3, phi_mid), step, 0.4).all()
    assert not v.grid_mask(mid, np.full(3, phi_mid), step, 0.4).any()
    # phi lines at multiples of step, theta mid-cell
    assert v.grid_mask(np.deg2rad(85.0), np.deg2rad(40.0), step, 0.4)
    assert not v.grid_mask(np.deg2rad(85.0), np.deg2rad(45.0), step, 0.4)


def test_grid_drawn_black_over_sky():
    theta = np.full((2, 2), np.deg2rad(30.0))  # on a theta line
    rgb = v.compose_rgb(
        np.zeros((2, 2), int),
        theta,
        full(1.0, 2),
        spec(background="two_color", grid=True),
    )
    assert np.array_equal(rgb[0, 0], v.GRID_COLOR)


def test_image_sampling_from_synthetic_png(tmp_path):
    import matplotlib.pyplot as plt

    img = np.zeros((2, 2, 3))
    img[0, 0], img[0, 1], img[1, 0], img[1, 1] = v.RED, v.GREEN, v.YELLOW, v.BLUE
    p = tmp_path / "sky.png"
    plt.imsave(p, img)
    sky = v.load_sky_image(p)
    assert sky.shape == (2, 2, 3) and sky.min() >= 0 and sky.max() <= 1
    s = spec(background="image", sky=sky)
    # theta<pi/2 -> top row; rel = phi - pi in [-pi, 0) -> left column, [0, pi) -> right
    top_left = v.sky_rgb(np.array(0.3), np.array(np.pi - 1.0), s)
    top_right = v.sky_rgb(np.array(0.3), np.array(np.pi + 1.0), s)
    bot_right = v.sky_rgb(np.array(2.8), np.array(np.pi + 1.0), s)
    assert np.allclose(top_left, sky[0, 0]) and np.allclose(top_right, sky[0, 1])
    assert np.allclose(bot_right, sky[1, 1])
    # extreme angles stay in bounds
    assert v.sky_rgb(np.array(np.pi), np.array(0.0), s).shape == (3,)


def test_image_background_requires_sky():
    with pytest.raises(ValueError):
        v.sky_rgb(np.array(1.0), np.array(1.0), spec(background="image"))


def test_load_sky_image_greyscale_and_alpha(tmp_path):
    import matplotlib.pyplot as plt

    rgba = np.ones((3, 3, 4)) * 0.5
    p = tmp_path / "a.png"
    plt.imsave(p, rgba)
    assert v.load_sky_image(p).shape == (3, 3, 3)


def test_sky_directions_kerr_and_weyl():
    class Cfg:
        half_angle = 0.5

    status = np.array([[0, 1], [0, 0]])
    kerr = {
        "status": status,
        "coord1": full(10.0, 2),
        "coord2": full(1.0, 2),
        "phi": full(7.0, 2),
    }
    th, ph = v.sky_directions(kerr, Cfg)
    assert np.isnan(th[0, 1]) and th[0, 0] == 1.0
    assert np.isclose(ph[0, 0], 7.0 - 2 * np.pi)
    weyl = {"status": status, "coord1": full(1.0, 2), "coord2": full(1.0, 2)}
    th, ph = v.sky_directions(weyl, Cfg)
    assert np.isclose(th[0, 0], np.pi - np.pi / 4) and np.isnan(ph[0, 1])
    assert 0 <= ph[0, 0] < 2 * np.pi


@pytest.mark.parametrize("bg", ["black_hole", "two_color", "greyscale", "four_color"])
def test_output_shape_and_range(bg):
    rng = np.random.default_rng(0)
    status = rng.integers(0, 4, (8, 8))
    theta = rng.uniform(0, np.pi, (8, 8))
    phi = rng.uniform(0, 2 * np.pi, (8, 8))
    rgb = v.compose_rgb(status, theta, phi, spec(background=bg, grid=bg != "black_hole"))
    assert rgb.shape == (8, 8, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
