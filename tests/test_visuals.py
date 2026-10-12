"""Pure visuals tests on synthetic arrays (no tracing)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from bh_shadows.plots import visuals as v

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


def test_light_weight_peak_and_falloff():
    tb = np.pi - HALF_PI
    peak = v.light_weight(np.array(tb), np.array(np.pi), tb, 10.0, 0.7)
    assert peak == pytest.approx(0.7)
    far = v.light_weight(np.array(tb + np.pi / 2), np.array(np.pi), tb, 10.0, 0.7)
    assert far < 1e-3
    ds = np.deg2rad([0, 5, 10, 20, 40])
    w = v.light_weight(HALF_PI + ds, np.full(5, np.pi), HALF_PI, 10.0, 0.7)
    assert np.all(np.diff(w) < 0)


def test_light_weight_nan_is_zero_without_warning(recwarn):
    nan = np.array([np.nan, 1.0])
    w = v.light_weight(nan, nan[::-1], HALF_PI, 10.0, 0.7)
    assert np.array_equal(w, [0.0, 0.0]) and not recwarn.list


def test_light_source_off_is_unchanged():
    args = (np.zeros((2, 2), int), full(1.0, 2), full(2.0, 2))
    base = v.compose_rgb(*args, spec(background="two_color"))
    off = v.compose_rgb(*args, spec(background="two_color", light_source=False))
    assert np.array_equal(base, off)


def test_light_blends_white_over_sky_and_grid_stays_on_top():
    tb = np.pi - HALF_PI
    st = np.zeros((2, 2), int)
    rgb = v.compose_rgb(
        st, np.full((2, 2), tb), full(np.pi, 2), spec(background="two_color", light_source=True)
    )
    assert np.allclose(rgb[0, 0], 0.7 * v.WHITE + 0.3 * v.BLUE)
    theta = np.full((2, 2), np.deg2rad(90.0))  # on a theta grid line, at the glow centre
    rgb = v.compose_rgb(
        st, theta, full(np.pi, 2), spec(background="two_color", light_source=True, grid=True)
    )
    assert np.array_equal(rgb[0, 0], v.GRID_COLOR)


def test_light_theta_obs_moves_peak():
    ob = np.deg2rad(60.0)
    s = spec(background="two_color", light_source=True, theta_obs=ob)
    th = np.array([[np.deg2rad(120.0), np.deg2rad(60.0)]])
    rgb = v.compose_rgb(np.zeros((1, 2), int), th, np.full((1, 2), np.pi), s)
    assert np.allclose(rgb[0, 0], 0.7 * v.WHITE + 0.3 * v.BLUE)
    assert not np.allclose(rgb[0, 1], 0.7 * v.WHITE + 0.3 * v.RED)


def test_from_config_theta_obs_weyl():
    from bh_shadows.runner.shadow_config import ShadowConfig

    cfg = ShadowConfig(M=0.9, MD=0.1)
    assert v.VisualSpec.from_config(cfg).theta_obs == pytest.approx(HALF_PI)
    kerr = ShadowConfig(a=0.9, theta0=60.0)
    assert v.VisualSpec.from_config(kerr).theta_obs == pytest.approx(np.deg2rad(60.0))


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
        backend = "kerr"

    class WeylCfg:
        half_angle = 0.5
        backend = "weyl"

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
    th, ph = v.sky_directions(weyl, WeylCfg)  # legacy dict without phi: flat-sky fallback
    assert np.isclose(th[0, 0], np.pi - np.pi / 4) and np.isnan(ph[0, 1])
    assert 0 <= ph[0, 0] < 2 * np.pi
    traced = weyl | {"phi": full(7.0, 2)}
    th, ph = v.sky_directions(traced, WeylCfg)
    assert np.isclose(th[0, 0], np.pi - np.pi / 4)
    assert np.isclose(ph[0, 0], 7.0 - 2 * np.pi)


def test_flat_end_angles_recover_pixel_angles():
    d, R = 15.0, 30.0
    a, b = np.meshgrid(np.linspace(-0.5, 0.5, 9), np.linspace(-0.5, 0.5, 9), indexing="ij")
    ux, uy, uz = -np.cos(a) * np.cos(b), -np.cos(a) * np.sin(b), np.sin(a)
    t = -d * ux + np.sqrt((d * ux) ** 2 - d**2 + R**2)
    x, y, z = d + t * ux, t * uy, t * uz
    ae, be = v.flat_end_angles(np.hypot(x, y), -z, np.arctan2(y, x), d)
    np.testing.assert_allclose(ae, a, atol=1e-12)
    np.testing.assert_allclose(be, b, atol=1e-12)
    step = 10.0
    m = v.grid_mask_flat_angle(np.hypot(x, y), -z, np.arctan2(y, x), d, step)
    assert m[np.abs(a) <= v.FLAT_ANGLE_HALF_WIDTH].all()  # alpha = 0 line is a full row band
    assert not v.grid_mask_flat_angle(*(np.array([np.nan]),) * 3, d, step).any()


def test_flat_end_angles_off_plane_observer():
    d, R, z0 = 15.0, 30.0, 0.07
    a, b = np.meshgrid(np.linspace(-0.5, 0.5, 9), np.linspace(-0.5, 0.5, 9), indexing="ij")
    ux, uy, uz = -np.cos(a) * np.cos(b), -np.cos(a) * np.sin(b), np.sin(a)
    t = -d * ux + np.sqrt((d * ux) ** 2 - d**2 + R**2)
    x, y, z = d + t * ux, t * uy, z0 - t * uz  # stored z: observer height, alpha sign flipped
    ae, be = v.flat_end_angles(np.hypot(x, y), z, np.arctan2(y, x), d, z0)
    np.testing.assert_allclose(ae, a, atol=1e-12)
    np.testing.assert_allclose(be, b, atol=1e-12)
    ae0, _ = v.flat_end_angles(np.hypot(x, y), z, np.arctan2(y, x), d)
    assert not np.allclose(ae0, a, atol=1e-6)


def test_compose_rgb_flat_angle_needs_rho_end_and_z_end():
    st = np.zeros((2, 2), int)
    ang = np.full((2, 2), 1.0)
    sp = spec(background="four_color", grid=True, grid_mode="flat_angle")
    with pytest.raises(ValueError, match="rho_end and z_end"):
        v.compose_rgb(st, ang, ang, sp, z_end=np.zeros((2, 2)))
    rgb = v.compose_rgb(st, ang, ang, sp, z_end=np.zeros((2, 2)), rho_end=np.full((2, 2), 30.0))
    assert (rgb == v.GRID_COLOR).all()  # z_end = 0 lies on the alpha_e = 0 line


@pytest.mark.parametrize("bg", ["black_hole", "two_color", "greyscale", "four_color"])
def test_output_shape_and_range(bg):
    rng = np.random.default_rng(0)
    status = rng.integers(0, 4, (8, 8))
    theta = rng.uniform(0, np.pi, (8, 8))
    phi = rng.uniform(0, 2 * np.pi, (8, 8))
    rgb = v.compose_rgb(status, theta, phi, spec(background=bg, grid=bg != "black_hole"))
    assert rgb.shape == (8, 8, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
