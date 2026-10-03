"""Tiny-grid end-to-end checks. One numba compile per interpreter; Mat_nu is set once."""

import json
import re

import numpy as np
import pytest

import generate_matriz as gm
import run_shadow as rs
import shadow_config as sc
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
