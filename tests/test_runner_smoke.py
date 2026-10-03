"""Tiny-grid end-to-end checks. One numba compile per interpreter; Mat_nu is set once."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

import generate_matriz as gm
import run_shadow as rs
import shadow_config as sc
import visuals as vis
import weyl_core as wc


@pytest.fixture(autouse=True)
def tmp_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(gm, "MATRICES_DIR", tmp_path / "matrices")
    monkeypatch.setattr(sc, "RUNS_DIR", tmp_path / "runs")


@pytest.mark.parametrize(
    "cfg_file", ["schwarzschild_nodisk", "schwarzschild_disk", "kerr_a0.9_nodisk"]
)
def test_shipped_configs_smoke(cfg_file):
    ov = {"n": 8, "n_grid": 20}
    if cfg_file == "schwarzschild_disk":
        ov["source_opt"] = "analytic"  # keeps the table cheap; physics path is the same
        ov["MD"] = 0.0
        ov["disk_visible_opt"] = True
    cfg = sc.ShadowConfig.from_toml(f"configs/{cfg_file}.toml", ov)
    out = rs.trace_grid(cfg)
    rs.save_run(cfg, out, elapsed_s=1.0, config_path=f"configs/{cfg_file}.toml")
    assert re.match(r"^\d{8}_\d{6}_", cfg.run_dir.name)
    meta = json.loads((cfg.run_dir / "run.json").read_text())
    assert meta["timestamp"] == cfg.timestamp
    assert meta["n"] == 8
    assert {"a", "M", "name", "run_id"} <= meta.keys()
    assert out["status"].shape == (8, 8)
    assert not (out["status"] == 3).any()
    assert (cfg.run_dir / "image.png").exists()
    assert (cfg.run_dir / "config.resolved.toml").exists()


def test_chunked_parallel_matches_single_call():
    cfg = sc.ShadowConfig.from_toml("configs/kerr_a0.9_nodisk.toml", {"n": 8, "parallel": True})
    one = rs.trace_grid(cfg, progress=False, chunk=1)
    whole = rs.trace_grid(cfg, progress=False, chunk=cfg.n)
    assert one.keys() == whole.keys()
    for k in one:
        assert np.array_equal(one[k], whole[k], equal_nan=True)


def test_func_disk_off_matches_wrapper_on_nodisk_rays():
    """With disk_visible=False the ray ignores the disk; the wrapper passes True."""
    M, MD, b, z0, hder = 1.0, 0.0, 6.0, 0.0, 1e-6
    wc.load_matrix(str(gm.ensure_matrix(M, MD, b, 20)))
    rho0 = wc.solve_observer_rho(M, MD, b, z0=z0)
    for alfa, beta in [(-0.4, 0.4), (-0.05, 0.05)]:
        y = np.array(
            [rho0, wc.dr(rho0, z0, M, MD, b, alfa, beta), z0, wc.dthe(rho0, z0, M, MD, b, alfa)]
        )
        off = wc.func(y, 300.0, -0.02, alfa, beta, M, rho0, z0, MD, b, hder, False)
        on = wc.func_bh_disk_crossing(y, 300.0, -0.02, alfa, beta, M, rho0, z0, MD, b, hder)
        if on[1] < 40.0:  # no disk hit: both modes must agree exactly
            assert list(on) == list(off)
        else:
            assert on[1] > 49.0


EXAMPLES = sorted(Path("configs/examples").glob("*.toml"))
TINY = {"n": 8, "n_grid": 20, "MD": 0.0, "source_opt": "analytic", "parallel": False, "show": False}


def _tiny(path, **extra):
    ov = dict(TINY) | extra
    if "disk" in path.name or "greyscale" in path.name or "two_color" in path.name:
        ov["disk_visible_opt"] = True  # MD=0 override: keep the disk branch, cheap matrix
    return sc.ShadowConfig.from_toml(path, ov)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example_configs_render_image(path):
    cfg = _tiny(path)
    if cfg.backend == "kerr":
        cfg = sc.ShadowConfig.from_toml(path, {"n": 8, "parallel": False})
    out = rs.trace_grid(cfg, progress=False)
    rs.save_run(cfg, out, config_path=str(path))
    assert (cfg.run_dir / "image.png").stat().st_size > 0


@pytest.mark.parametrize("cfg_file", ["schwarzschild_nodisk", "kerr_a0.9_nodisk"])
def test_four_color_orientation(cfg_file, tmp_path):
    cfg = sc.ShadowConfig.from_toml(
        f"configs/{cfg_file}.toml",
        {"n": 16, "n_grid": 20, "background": "four_color", "parallel": False},
    )
    out = rs.trace_grid(cfg, progress=False)
    theta, phi = vis.sky_directions(out, cfg)
    rgb = vis.compose_rgb(out["status"], theta, phi, vis.VisualSpec.from_config(cfg))
    # imshow origin="lower": last row is the top, column 0 the left
    assert out["status"][-1, 0] == 0 and out["status"][0, -1] == 0
    assert np.array_equal(rgb[-1, 0], vis.RED)  # top-left
    assert np.array_equal(rgb[0, -1], vis.BLUE)  # bottom-right
    rs.save_run(cfg, out)
    assert (cfg.run_dir / "image.png").exists()


def test_palette_still_uses_plot_status(monkeypatch):
    calls = []
    monkeypatch.setattr(rs, "plot_status", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(rs.visuals, "render_image", lambda *a, **k: calls.append("visuals"))
    cfg = sc.ShadowConfig.from_toml("configs/schwarzschild_nodisk.toml", {"n": 8})
    out = {"status": np.zeros((8, 8), int), "coord1": np.zeros((8, 8)), "coord2": np.zeros((8, 8))}
    rs.save_run(cfg, out)
    assert len(calls) == 1 and calls[0][0] is out["status"]
    cfg2 = sc.ShadowConfig.from_toml(
        "configs/schwarzschild_nodisk.toml", {"n": 8, "background": "two_color"}
    )
    rs.save_run(cfg2, out)
    assert calls[-1] == "visuals" and len(calls) == 2
