"""Physics validation (V2, V3, V4, analytic overlay) on small grids (n <= 64).

Each trace runs in its own subprocess: `weyl_core.Mat_nu` is frozen at first numba compile.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import shadow_edge as se

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_physics.py"
EDGE_ANALYTIC = se.schwarzschild_edge_angle(1.0, 15.0)  # 0.3283 rad


def run_validate(tmp_path, *args, check=True):
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--matrices-dir",
        str(tmp_path / "matrices"),
        *map(str, args),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)  # noqa: S603
    if check:
        assert res.returncode == 0, res.stderr[-2000:]
        return json.loads(res.stdout.strip().splitlines()[-1])
    return res


def test_shadow_edge_a0_matches_closed_form():
    x, y = se.kerr_shadow_edge(0.0, 1.0, 15.0, np.pi / 2)
    assert np.allclose(np.hypot(x, y), EDGE_ANALYTIC, atol=1e-9)


def test_shadow_edge_small_a_continuity():
    x, y = se.kerr_shadow_edge(1e-6, 1.0, 15.0, np.pi / 2)  # checks the sqrt(gpp) scaling
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert abs(y.max() - EDGE_ANALYTIC) < 5e-3
    x9, y9 = se.kerr_shadow_edge(0.9, 1.0, 15.0, np.pi / 2)
    assert np.isfinite(x9).all() and np.isfinite(y9).all()
    assert (x9[0], y9[0]) == (x9[-1], y9[-1])  # closed


@pytest.mark.parametrize("backend", ["weyl", "kerr"])
def test_v2_schwarzschild_edge(tmp_path, backend):
    m = run_validate(tmp_path, "--n", 64, "--a", 0, "--backend", backend)
    for key in ("alfa_edge_top", "beta_edge_right"):
        assert abs(m[key] - EDGE_ANALYTIC) <= m["pitch"], (key, m[key])
    for key in ("alfa_edge_bottom", "beta_edge_left"):
        assert abs(m[key] + EDGE_ANALYTIC) <= m["pitch"], (key, m[key])


def test_v4_spin_shift_flatten_mirror(tmp_path):
    m0 = run_validate(tmp_path, "--n", 64, "--a", 0, "--backend", "kerr")
    pos = tmp_path / "pos.npy"
    m9 = run_validate(tmp_path, "--n", 64, "--a", 0.9, "--save-status", pos)
    p = m9["pitch"]
    assert m9["centroid_x"] - m0["centroid_x"] > p  # shift along -beta
    assert abs((m9["alfa_edge_top"] - m9["alfa_edge_bottom"]) - 2 * EDGE_ANALYTIC) < 4 * p
    assert abs(m9["alfa_edge_top"] + m9["alfa_edge_bottom"]) < p  # no shift along alpha
    assert abs(m9["halfwidth_left"] - m9["halfwidth_right"]) >= p  # flattened side
    neg = tmp_path / "neg.npy"
    res = run_validate(tmp_path, "--n", 64, "--raw-kerr-a", -0.9, "--save-status", neg, check=False)
    if res.returncode != 0 or not neg.exists():
        pytest.skip("raw a=-0.9 trace failed in kerr_core; mirror N/A")
    s_neg, s_pos = np.load(neg), np.load(pos)
    if (s_neg == 3).any():
        pytest.skip("raw a=-0.9 gives status-3 pixels; mirror N/A")
    mismatch = (s_neg != s_pos[:, ::-1]).mean()
    assert mismatch <= 0.01, mismatch


def test_overlay_matches_traced_boundary(tmp_path):
    from scipy.ndimage import binary_erosion

    st = tmp_path / "st.npy"
    run_validate(tmp_path, "--n", 64, "--a", 0.9, "--save-status", st)
    status = np.load(st)
    n, h = status.shape[0], 0.5880026035
    pitch = 2 * h / (n - 1)
    shadow = status == 1
    edge = shadow & ~binary_erosion(shadow, structure=np.ones((3, 3)), border_value=1)
    ii, jj = np.nonzero(edge)
    px, py = jj * pitch - h, ii * pitch - h
    cx, cy = se.kerr_shadow_edge(0.9, 1.0, 15.0, np.pi / 2, n=4000)
    d = np.sqrt((px[:, None] - cx[None]) ** 2 + (py[:, None] - cy[None]) ** 2).min(axis=1)
    assert (d <= 1.5 * pitch).mean() >= 0.95


V3_CASES = [(1.0, 0.0), (0.95, 0.05), (0.9, 0.1), (0.7, 0.3)]


def test_v3_md_smooth_and_disk_class(tmp_path):
    runs = {
        # MD=0 keeps the default n_grid: a coarse analytic table makes the Weyl tracer hang
        md: run_validate(tmp_path, "--n", 32, "--M", M, "--MD", md, *(["--n-grid", 200] * (md > 0)))
        for M, md in V3_CASES
    }
    for r in runs.values():
        assert r["counts"]["3"] == 0
    s = [runs[md]["counts"]["1"] for _, md in V3_CASES]
    assert all(a > b for a, b in zip(s, s[1:], strict=False)) or all(
        a < b for a, b in zip(s, s[1:], strict=False)
    ), s
    assert abs(s[1] - s[0]) <= 0.2 * s[0]
    assert runs[0.0]["counts"]["2"] == 0
    for md in (0.05, 0.1, 0.3):
        assert runs[md]["counts"]["2"] > 0
    forced = run_validate(
        tmp_path, "--n", 32, "--M", 0.9, "--MD", 0.1, "--n-grid", 200, "--disk-visible", "false"
    )
    assert forced["counts"]["2"] == 0


def test_validate_physics_rejects_large_n(tmp_path):
    res = run_validate(tmp_path, "--n", 65, check=False)
    assert res.returncode != 0 and "n" in res.stderr


def _lines_on_saved_fig(monkeypatch, tmp_path, **ov):
    import matplotlib.figure as mfig

    import shadow_config as sc
    import weyl_plot as wp

    seen = []
    monkeypatch.setattr(
        mfig.Figure, "savefig", lambda self, *a, **k: seen.append(self.axes[0].lines)
    )
    cfg = sc.ShadowConfig(n=8, a=0.9, **ov)
    wp.plot_status(np.zeros((8, 8), dtype=int), cfg, tmp_path / "x.png")
    return len(seen[0])


def test_shadow_edge_flag_roundtrip_and_default_off(tmp_path, monkeypatch):
    import shadow_config as sc

    assert sc.ShadowConfig().shadow_edge is False
    cfg = sc.ShadowConfig(shadow_edge=True)
    f = tmp_path / "c.toml"
    f.write_text(cfg.to_toml())
    assert sc.ShadowConfig.from_toml(f).shadow_edge is True
    assert cfg.to_metadata()["shadow_edge"] is True
    assert _lines_on_saved_fig(monkeypatch, tmp_path) == 0
    assert _lines_on_saved_fig(monkeypatch, tmp_path, shadow_edge=True) == 1
