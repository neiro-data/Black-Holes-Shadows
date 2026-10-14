import dataclasses
import json
import logging

import numpy as np
import pytest

from bh_shadows.core import matrix as gm
from bh_shadows.runner.shadow_config import ShadowConfig


def write_cfg(tmp_path, body):
    p = tmp_path / "c.toml"
    p.write_text(body)
    return p


def test_overrides_and_routing(tmp_path):
    p = write_cfg(tmp_path, "[spacetime]\nM = 0.9\nMD = 0.1\n")
    cfg = ShadowConfig.from_toml(p)
    assert cfg.backend == "weyl" and cfg.disk_visible and cfg.matrix_source == "quadrature"
    cfg = ShadowConfig.from_toml(p, {"M": 1.0, "MD": 0.0})
    assert not cfg.disk_visible and cfg.matrix_source == "analytic"
    cfg = ShadowConfig.from_toml(p, {"M": 1.0, "MD": 0.0, "a": 0.9})
    assert cfg.backend == "kerr"


@pytest.mark.parametrize(
    "ov",
    [
        {"a": 1.0},
        {"a": -0.1},
        {"MD": -1.0},
        {"b": 0.0},
        {"n": 7},
        {"a": 0.5, "M": 0.9, "MD": 0.1},
        {"M": 0.5},
        {"palette": "nope"},
    ],
)
def test_validation_errors(tmp_path, ov):
    with pytest.raises(ValueError):
        ShadowConfig.from_toml(write_cfg(tmp_path, ""), ov)


def test_analytic_matches_quadrature():
    ana = gm.build_lambda_matrix(1.0, 0.0, 6.0, 20, "analytic")
    quad = gm.build_lambda_matrix(1.0, 0.0, 6.0, 20, "quadrature")
    assert np.max(np.abs(ana - quad)) < 1e-4


def test_matrix_sidecar_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(gm, "MATRICES_DIR", tmp_path)
    path = gm.ensure_matrix(1.0, 0.0, 6.0, 20)
    assert path.exists()
    path.with_suffix(".json").write_text(json.dumps({"M": 2.0}))
    with pytest.raises(ValueError):
        gm.ensure_matrix(1.0, 0.0, 6.0, 20)


def test_ensure_matrix_logs_build_then_cache(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(gm, "MATRICES_DIR", tmp_path)
    caplog.set_level(logging.INFO, logger="bh_shadows.core.matrix")
    gm.ensure_matrix(1.0, 0.0, 6.0, 20)
    assert "built in" in caplog.text
    caplog.clear()
    gm.ensure_matrix(1.0, 0.0, 6.0, 20)
    assert "cached" in caplog.text and "built in" not in caplog.text


def test_timestamp_validation_and_run_dir(tmp_path):
    cfg_file = write_cfg(tmp_path, 'name = "x"\n')
    with pytest.raises(ValueError):
        ShadowConfig.from_toml(cfg_file, {"timestamp": "bad"})
    cfg = ShadowConfig.from_toml(cfg_file, {"timestamp": "20260101_000000"})
    assert cfg.run_dir.name == "20260101_000000_x"


def _png(tmp_path, name="sky.png"):
    import matplotlib.pyplot as plt

    p = tmp_path / name
    plt.imsave(p, np.zeros((2, 2, 3)))
    return p


def test_plot_defaults_equal_legacy(tmp_path):
    cfg = ShadowConfig.from_toml(write_cfg(tmp_path, ""))
    assert cfg.background == "palette" and cfg.grid is False
    assert (cfg.grid_step_deg, cfg.grid_width_deg, cfg.disk_alpha) == (10.0, 0.4, 0.6)
    assert cfg.image_path == ""
    assert cfg.grid_mode == "theta"


@pytest.mark.parametrize(
    "ov",
    [
        {"background": "nope"},
        {"grid": True, "background": "black_hole"},
        {"grid": True},  # palette
        {"background": "image"},
        {"background": "image", "image_path": "/nonexistent/sky.png"},
        {"disk_alpha": -0.1},
        {"disk_alpha": 1.1},
        {"grid_step_deg": 0.0},
        {"grid_step_deg": 91.0},
        {"grid_width_deg": 0.0},
        {"grid_step_deg": 10.0, "grid_width_deg": 5.0},
        {"grid_mode": "nope"},
        {
            "grid": True,
            "background": "four_color",
            "grid_mode": "flat_angle",
            "backend_opt": "kerr",
        },
    ],
)
def test_plot_validation_errors(tmp_path, ov):
    with pytest.raises(ValueError):
        ShadowConfig.from_toml(write_cfg(tmp_path, ""), ov)


def test_disk_alpha_bounds_accepted(tmp_path):
    for alpha in (0.0, 1.0):
        assert ShadowConfig.from_toml(write_cfg(tmp_path, ""), {"disk_alpha": alpha})


@pytest.mark.parametrize("bg", ["palette", "black_hole", "greyscale"])
def test_light_source_needs_two_or_four_color(tmp_path, bg):
    with pytest.raises(ValueError, match="light_source"):
        ShadowConfig.from_toml(write_cfg(tmp_path, ""), {"background": bg, "light_source": True})


def test_light_source_with_image_raises(tmp_path):
    png = _png(tmp_path)
    with pytest.raises(ValueError, match="light_source"):
        ShadowConfig.from_toml(
            write_cfg(tmp_path, ""),
            {"background": "image", "image_path": str(png), "light_source": True},
        )


@pytest.mark.parametrize(
    "key,val", [("light_alpha", -0.1), ("light_alpha", 1.1), ("light_radius_deg", 0)]
)
def test_light_params_rejected(tmp_path, key, val):
    with pytest.raises(ValueError, match=key):
        ShadowConfig.from_toml(write_cfg(tmp_path, ""), {key: val})


def test_light_bounds_accepted(tmp_path):
    for alpha in (0.0, 1.0):
        assert ShadowConfig.from_toml(write_cfg(tmp_path, ""), {"light_alpha": alpha})


def test_light_keys_roundtrip_and_run_json(tmp_path):
    body = (
        '[plot]\nbackground = "four_color"\nlight_source = true\n'
        "light_alpha = 0.4\nlight_radius_deg = 7.5\n"
    )
    cfg = ShadowConfig.from_toml(write_cfg(tmp_path, body))
    back = ShadowConfig.from_toml(write_cfg(tmp_path, cfg.to_toml()))
    got = (back.light_source, back.light_alpha, back.light_radius_deg)
    assert got == (True, 0.4, 7.5)
    meta = cfg.to_metadata()
    assert (meta["light_source"], meta["light_alpha"], meta["light_radius_deg"]) == (True, 0.4, 7.5)


def test_relative_image_resolves_against_config_dir(tmp_path):
    sub = tmp_path / "cfgs"
    sub.mkdir()
    png = _png(sub)
    body = '[plot]\nbackground = "image"\nimage = "sky.png"\n'
    p = sub / "c.toml"
    p.write_text(body)
    cfg = ShadowConfig.from_toml(p)
    assert cfg.image_path == str(png.resolve())


def test_missing_image_key_with_image_background_raises(tmp_path):
    with pytest.raises(ValueError):
        ShadowConfig.from_toml(write_cfg(tmp_path, '[plot]\nbackground = "image"\n'))


def test_plot_to_toml_round_trip(tmp_path):
    png = _png(tmp_path)
    src = write_cfg(
        tmp_path,
        '[plot]\nbackground = "image"\nimage = "sky.png"\ngrid = true\n'
        "grid_step_deg = 15.0\ngrid_width_deg = 0.5\ndisk_alpha = 0.25\n",
    )
    cfg = ShadowConfig.from_toml(src, {"timestamp": "20260101_000000"})
    out = tmp_path / "round.toml"
    out.write_text(cfg.to_toml())
    back = ShadowConfig.from_toml(out, {"timestamp": "20260101_000000"})
    assert back == dataclasses.replace(cfg, name=cfg.run_name)  # to_toml pins the run name
    assert back.image_path == str(png.resolve())


def test_palette_written_only_for_palette_background(tmp_path):
    pal = ShadowConfig.from_toml(write_cfg(tmp_path, ""), {"palette": "quadrant"})
    assert 'palette = "quadrant"' in pal.to_toml()
    assert pal.to_metadata()["palette"] == "quadrant"
    four = dataclasses.replace(pal, background="four_color")
    assert "palette" not in four.to_toml()
    assert four.to_metadata()["palette"] is None
    back = ShadowConfig.from_toml(write_cfg(tmp_path, four.to_toml()))
    assert back.background == "four_color"


def test_grid_mode_flat_angle_roundtrip_and_weyl_only(tmp_path):
    ov = {"grid": True, "background": "four_color", "grid_mode": "flat_angle"}
    cfg = ShadowConfig.from_toml(write_cfg(tmp_path, ""), ov)
    assert cfg.backend == "weyl" and cfg.grid_mode == "flat_angle"
    again = tmp_path / "again.toml"
    again.write_text(cfg.to_toml())
    assert ShadowConfig.from_toml(again).grid_mode == "flat_angle"
    assert cfg.to_metadata()["grid_mode"] == "flat_angle"
    with pytest.raises(ValueError, match="Weyl-only"):
        ShadowConfig.from_toml(write_cfg(tmp_path, ""), ov | {"a": 0.5})


@pytest.mark.parametrize("ov", [{"a": 0.5}, {"M": 1.0, "MD": 0.0, "backend_opt": "kerr"}])
def test_off_plane_rejected_for_kerr(tmp_path, ov):
    with pytest.raises(ValueError, match="off_plane"):
        ShadowConfig.from_toml(write_cfg(tmp_path, ""), {"off_plane": True} | ov)


def test_off_plane_accepted_without_disk(tmp_path):
    cfg = ShadowConfig.from_toml(write_cfg(tmp_path, ""), {"off_plane": True})
    assert cfg.backend == "weyl" and cfg.MD == 0 and cfg.off_plane


def test_off_plane_accepted_and_roundtrips(tmp_path):
    p = write_cfg(tmp_path, "[spacetime]\nM = 0.9\nMD = 0.1\n[observer]\noff_plane = true\n")
    cfg = ShadowConfig.from_toml(p)
    assert cfg.off_plane and "off_plane = true" in cfg.to_toml()
    q = write_cfg(tmp_path, cfg.to_toml())
    assert ShadowConfig.from_toml(q).off_plane
    assert cfg.to_metadata()["off_plane"] is True


def test_observer_z_default_and_z0(tmp_path):
    cfg = ShadowConfig.from_toml(write_cfg(tmp_path, ""))
    assert cfg.observer_z == 0.0
    cfg = ShadowConfig.from_toml(write_cfg(tmp_path, ""), {"off_plane": True})
    assert cfg.observer_z == 0.07
    cfg = ShadowConfig.from_toml(write_cfg(tmp_path, ""), {"off_plane": True, "z0": 1.0})
    assert cfg.observer_z == 1.0


@pytest.mark.parametrize(
    "ov",
    [
        {"z0": 1.0, "a": 0.5, "off_plane": True},  # Kerr
        {"z0": 1.0},  # no off_plane
        {"z0": 15.0, "off_plane": True},  # |z0| >= distance
        {"z0": float("nan"), "off_plane": True},
        {"disk_outer": 6.0},  # <= b
        {"disk_outer": 12.0, "a": 0.5},  # finite with Kerr
        {"background": "shaded_disk", "a": 0.5},
        {"background": "shaded_disk", "grid": True},
    ],
)
def test_z0_disk_outer_shaded_disk_errors(tmp_path, ov):
    with pytest.raises(ValueError):
        ShadowConfig.from_toml(write_cfg(tmp_path, ""), ov)


@pytest.mark.parametrize("outer", [12.0, float("inf")])
def test_z0_disk_outer_roundtrip_and_run_json(tmp_path, outer):
    cfg = ShadowConfig.from_toml(
        write_cfg(tmp_path, ""),
        {"off_plane": True, "z0": 1.0, "disk_outer": outer, "background": "shaded_disk"},
    )
    back = ShadowConfig.from_toml(write_cfg(tmp_path, cfg.to_toml()))
    assert (back.z0, back.disk_outer, back.background) == (1.0, outer, "shaded_disk")
    meta = cfg.to_metadata()
    assert meta["z0"] == 1.0 and meta["observer_z"] == 1.0
    assert meta["disk_outer"] == (None if outer == float("inf") else outer)
    json.dumps(meta, allow_nan=False)
