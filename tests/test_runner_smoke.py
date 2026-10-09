"""Tiny-grid end-to-end checks. One numba compile per interpreter; Mat_nu is set once."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from bh_shadows.core import matrix as gm
from bh_shadows.core import weyl_core as wc
from bh_shadows.plots import visuals as vis
from bh_shadows.runner import run_shadow as rs
from bh_shadows.runner import shadow_config as sc


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
        ov["M"] = 1.0
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


EXAMPLES = sorted(Path("configs/examples").glob("*.toml"))
TINY = {
    "n": 8,
    "n_grid": 20,
    "M": 1.0,
    "MD": 0.0,
    "source_opt": "analytic",
    "parallel": False,
    "show": False,
}


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


@pytest.mark.parametrize("cfg_file", ["schwarzschild_nodisk", "kerr_a0.9_nodisk"])
def test_save_run_writes_npz(cfg_file):
    cfg = sc.ShadowConfig.from_toml(
        f"configs/{cfg_file}.toml", {"n": 8, "n_grid": 20, "parallel": False, "show": False}
    )
    out = rs.trace_grid(cfg, progress=False)
    rs.save_run(cfg, out)
    npz = np.load(cfg.run_dir / "outputs.npz")
    assert {"status", "coord1", "coord2"} <= set(npz.files)
    assert "phi" in npz.files
    assert npz["phi"].shape == npz["status"].shape
    assert np.array_equal(npz["status"], out["status"])
    assert not list(cfg.run_dir.glob("*.txt"))


def test_serial_path_validates_device():
    """A bad `device_opt` must raise even when parallel=false (post-init validation bypassed)."""
    cfg = sc.ShadowConfig.from_toml("configs/kerr_a0.9_nodisk.toml", {"n": 8, "parallel": False})
    object.__setattr__(cfg, "device_opt", "bogus")
    with pytest.raises(ValueError, match="device must be"):
        rs._resolve_run_device(cfg)
    object.__setattr__(cfg, "device_opt", "gpu")
    assert rs._resolve_run_device(cfg) == "cpu"


def _random_pixels(n, k=12, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, k), rng.integers(0, n, k)


def test_retrace_points_matches_f_paral_weyl():
    M, MD, b, z0, hder, n = 1.0, 0.0, 6.0, 0.0, 1e-6, 16
    wc.load_matrix(str(gm.ensure_matrix(M, MD, b, 20)))
    rho0 = wc.solve_observer_rho(M, MD, b, z0=z0)
    h = 0.05
    alpha, beta = np.linspace(-h, h, n), np.linspace(h, -h, n)
    ii, jj = _random_pixels(n)
    for disk in (False, True):
        full = wc.f_paral(rho0, z0, M, MD, b, alpha, beta, hder, disk)
        got = wc.retrace_points(rho0, z0, M, MD, b, alpha, beta, ii, jj, hder, disk)
        for f, g in zip(full, got, strict=True):
            assert np.array_equal(f[ii, jj], g)


def test_retrace_points_matches_f_paral_kerr():
    from bh_shadows.core import kerr_core as kc

    r0, theta0, a, M, n = 50.0, np.pi / 2, 0.9, 1.0, 16
    h = 0.1
    alpha, beta = np.linspace(-h, h, n), np.linspace(h, -h, n)
    ii, jj = _random_pixels(n)
    full = kc.f_paral(r0, theta0, a, M, alpha, beta)
    got = kc.retrace_points(r0, theta0, a, M, alpha, beta, ii, jj)
    for f, g in zip(full, got, strict=True):
        assert np.array_equal(f[ii, jj], g, equal_nan=True)


def _weyl_full_phi(n=16):
    """Trace the full (unmirrored) Weyl grid and return it with the config and the run output."""
    cfg = sc.ShadowConfig.from_toml(
        "configs/schwarzschild_nodisk.toml", {"n": n, "n_grid": 20, "show": False}
    )
    out = rs.trace_grid(cfg, progress=False)
    rho0 = wc.solve_observer_rho(cfg.M, cfg.MD, cfg.b, z0=0.0, target_gpp_sqrt=cfg.distance)
    alpha, beta = rs._angles(cfg)
    full = wc.f_paral(rho0, 0.0, cfg.M, cfg.MD, cfg.b, alpha, beta, rs.HDER, cfg.disk_visible)
    return cfg, out, full, beta


def test_weyl_phi_finite_and_antisymmetric_in_beta():
    cfg, out, (Mat, Mz, Mphi), beta = _weyl_full_phi(16)
    esc = out["status"] == 0
    assert esc.any() and np.isfinite(Mphi[esc]).all()
    # beta -> -beta maps phi -> -phi (observer at phi0 = 0); mirrored output equals the full trace
    np.testing.assert_allclose(Mphi[:, ::-1][esc], -Mphi[esc], atol=1e-8)
    np.testing.assert_allclose(out["phi"][esc], Mphi[esc], atol=1e-8)


def test_weyl_phi_zero_on_beta_zero_column():
    cfg = sc.ShadowConfig.from_toml(
        "configs/schwarzschild_nodisk.toml", {"n": 8, "n_grid": 20, "show": False}
    )
    rs.trace_grid(cfg, progress=False)  # loads the matrix
    rho0 = wc.solve_observer_rho(cfg.M, cfg.MD, cfg.b, z0=0.0, target_gpp_sqrt=cfg.distance)
    alpha = np.linspace(-0.3, 0.3, 8)
    _, _, Mphi = wc.f_paral(rho0, 0.0, cfg.M, cfg.MD, cfg.b, alpha, np.array([0.0]), rs.HDER, False)
    np.testing.assert_allclose(Mphi, 0.0, atol=1e-10)


def test_weyl_phi_matches_kerr_a0():
    base = {"n": 32, "n_grid": 20, "show": False}
    weyl = rs.trace_grid(
        sc.ShadowConfig.from_toml("configs/schwarzschild_nodisk.toml", base), progress=False
    )
    kerr = rs.trace_grid(
        sc.ShadowConfig.from_toml(
            "configs/kerr_a0.9_nodisk.toml", base | {"a": 0.0, "backend_opt": "kerr"}
        ),
        progress=False,
    )
    both = (weyl["status"] == 0) & (kerr["status"] == 0)
    assert both.sum() > 100
    # Both escape surfaces differ slightly (Weyl rho/z vs Boyer-Lindquist r), so compare loosely;
    # rays that wind near the photon sphere differ most.
    diff = np.abs(weyl["phi"][both] - kerr["phi"][both])
    assert np.median(diff) < 1e-2
    assert np.mean(diff < 2e-2) > 0.8


def _off_plane_cfg(n=16, **extra):
    ov = {"n": n, "n_grid": 20, "show": False, "parallel": False, "M": 0.9, "MD": 0.1} | extra
    return sc.ShadowConfig.from_toml("configs/schwarzschild_disk.toml", ov)


def test_off_plane_at_zero_offset_matches_quadrant_path(monkeypatch):
    """Full-alpha trace at z0 = 0 reproduces the mirrored quadrant (settles the z sign, D4)."""
    ref = rs.trace_grid(_off_plane_cfg(), progress=False)
    monkeypatch.setattr(rs, "OBSERVER_Z_OFFSET", 0.0)
    got = rs.trace_grid(_off_plane_cfg(off_plane=True), progress=False)
    assert np.array_equal(ref["status"], got["status"])
    # only escaped pixels carry a sky direction (disk hits store z + 50; the old mirror flips it)
    keep = ref["status"] == 0
    for k in ("coord1", "coord2", "phi"):
        np.testing.assert_allclose(got[k][keep], ref[k][keep], atol=1e-8, equal_nan=True)


def test_off_plane_symmetry_and_asymmetry():
    cfg = _off_plane_cfg(off_plane=True)
    out = rs.trace_grid(cfg, progress=False)
    n = cfg.n
    assert out["status"].shape == (n, n)
    esc = out["status"] == 0
    assert esc.any() and np.isfinite(out["phi"][esc]).all()
    # column mirror: rho, z equal; phi antisymmetric about 0
    np.testing.assert_allclose(out["coord1"][:, ::-1], out["coord1"], equal_nan=True)
    np.testing.assert_allclose(out["coord2"][:, ::-1], out["coord2"], equal_nan=True)
    np.testing.assert_allclose(out["phi"][:, ::-1][esc], -out["phi"][esc], atol=1e-8)
    # the alpha mirror is lost: the observer sees more disk below than above
    assert not np.array_equal(esc, esc[::-1])
    # disk hits are not symmetric about the middle row
    disk = out["status"] == 2
    assert disk.any() and not np.array_equal(disk, disk[::-1])


def test_off_plane_default_is_off():
    assert sc.ShadowConfig().off_plane is False


def test_off_plane_without_disk_runs():
    cfg = sc.ShadowConfig.from_toml(
        "configs/schwarzschild_nodisk.toml",
        {"n": 16, "n_grid": 20, "show": False, "parallel": False, "off_plane": True},
    )
    out = rs.trace_grid(cfg, progress=False)
    esc = out["status"] == 0
    assert out["status"].shape == (16, 16) and esc.any() and np.isfinite(out["phi"][esc]).all()
    assert (out["status"] == 1).any()
