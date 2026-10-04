"""GPU backend: device resolution (no MLX needed) and CPU-vs-GPU agreement (skipped w/o MLX)."""

import numpy as np
import pytest

from bh_shadows.gpu import backend as gb
from bh_shadows.runner import run_shadow as rs
from bh_shadows.runner import shadow_config as sc


@pytest.fixture(autouse=True)
def tmp_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "RUNS_DIR", tmp_path / "runs")


def test_device_option_parses_and_validates():
    cfg = sc.ShadowConfig.from_toml("configs/kerr_a0.9_nodisk.toml", {"device_opt": "gpu"})
    assert cfg.device_opt == "gpu"
    assert sc.ShadowConfig().device_opt == "auto"
    assert 'device = "gpu"' in cfg.to_toml()
    with pytest.raises(ValueError, match="device"):
        sc.ShadowConfig(device_opt="tpu")


def test_cli_device_flag():
    _, ov = rs.parse_args(["configs/kerr_a0.9_nodisk.toml", "--device", "cpu"])
    assert ov["device_opt"] == "cpu"


def test_resolve_device_fallbacks(monkeypatch, caplog):
    monkeypatch.setattr(gb, "gpu_available", lambda: False)
    assert gb.resolve_device("cpu") == "cpu"
    assert gb.resolve_device("auto") == "cpu"
    with caplog.at_level("WARNING"):
        assert gb.resolve_device("gpu") == "cpu"
    assert "using cpu" in caplog.text
    with pytest.raises(ValueError):
        gb.resolve_device("tpu")


def test_auto_stays_cpu_even_with_gpu(monkeypatch):
    monkeypatch.setattr(gb, "gpu_available", lambda: True)
    assert gb.resolve_device("auto") == "cpu"
    assert gb.resolve_device("gpu") == "gpu"


def _gpu_or_skip():
    pytest.importorskip("mlx.core")
    if not gb.gpu_available():
        pytest.skip("no Apple GPU")


def test_kerr_geo_rhs_matches_cpu():
    _gpu_or_skip()
    from bh_shadows.core import kerr_core as kc
    from bh_shadows.gpu import kerr_gpu as kg

    r0, th0, a, M, al, be = 15.0, np.pi / 2, 0.9, 1.0, 0.1, 0.2
    states = np.array(
        [
            [10, 0.05, 1.5, 0.02, 0],
            [6, -0.1, 1.2, 0.05, 0.3],
            [4, 0.2, 1.8, -0.03, 1],
            [20, 0.1, 1.0, 0.01, 2],
            [3, -0.2, 1.4, 0.04, 0.5],
        ]
    )
    gpu = kg.geo_gpu(states, r0, th0, a, M, al, be)
    cpu = np.array([kc.geo(0, s, a, M, al, be, r0, th0) for s in states])
    np.testing.assert_allclose(gpu, cpu, rtol=1e-4, atol=1e-6)


def test_kerr_gpu_agrees_with_cpu():
    _gpu_or_skip()
    base = {"n": 64, "name": "agree"}
    cpu = rs.trace_grid(
        sc.ShadowConfig.from_toml("configs/kerr_a0.9_nodisk.toml", base | {"device_opt": "cpu"}),
        progress=False,
    )
    cfg = sc.ShadowConfig.from_toml("configs/kerr_a0.9_nodisk.toml", base | {"device_opt": "gpu"})
    gpu = rs.trace_grid(cfg, progress=False)
    assert rs.RUN_STATS["device"] == "gpu"

    sc_, sg = cpu["status"], gpu["status"]
    mism = sc_ != sg
    assert mism.sum() <= 0.01 * sc_.size
    # every mismatch lies within 1 pixel of a CPU shadow boundary
    shadow = sc_ == 1
    pad = np.pad(shadow, 1, mode="edge")
    near = np.zeros_like(shadow)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            window = pad[1 + di : 1 + di + shadow.shape[0], 1 + dj : 1 + dj + shadow.shape[1]]
            near |= window != shadow
    assert not (mism & ~near).any()
    both_escape = (sc_ == 0) & (sg == 0)
    assert np.median(np.abs(cpu["coord2"][both_escape] - gpu["coord2"][both_escape])) < 1e-3


# ---------------------------------------------------------------------------
# Weyl GPU tracer
# ---------------------------------------------------------------------------
from bh_shadows.core import matrix as gm  # noqa: E402
from bh_shadows.core import weyl_core as wc  # noqa: E402

WEYL_N_GRID = 20
WEYL_CASES = [
    pytest.param(0.0, id="nodisk"),
    pytest.param(0.1, id="disk"),
]


def _seed_matrix(tmp_path, monkeypatch, M, MD, b):
    """Write the MD=0 analytic table under the (M, MD, b) name.

    numba freezes `Mat_nu` at first compile, so every test in the process must see the same
    table content. The kernels only treat it as lookup data, so CPU/GPU comparisons stay valid.
    """
    import json

    monkeypatch.setattr(gm, "MATRICES_DIR", tmp_path / "matrices")
    path = gm.matrix_path(M, MD, b, WEYL_N_GRID)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, lam=gm.build_lambda_matrix(M, 0.0, b, WEYL_N_GRID, "analytic"))
    path.with_suffix(".json").write_text(
        json.dumps({"M": M, "MD": MD, "b": b, "n_grid": WEYL_N_GRID})
    )


def _weyl_setup(tmp_path, monkeypatch, MD, M=1.0, b=6.0):
    _seed_matrix(tmp_path, monkeypatch, M, MD, b)
    wc.load_matrix(str(gm.ensure_matrix(M, MD, b, WEYL_N_GRID)))
    return M, b, wc.Mat_nu


def _frozen_or_skip(lam):
    """The CPU reference reads the table numba froze; skip if another one was frozen first."""
    p = np.array([[10.0, 1.0], [20.0, -7.0], [5.0, 12.0]])
    i = (40 - p[:, 1]) / 40 * (lam.shape[0] - 1) / 2
    j = (40 - p[:, 0]) / 40 * (lam.shape[1] - 1)
    i0, j0 = np.floor(i).astype(int), np.floor(j).astype(int)
    fi, fj = i - i0, j - j0
    f00, f01, f10, f11 = lam[i0, j0], lam[i0, j0 + 1], lam[i0 + 1, j0], lam[i0 + 1, j0 + 1]
    ref = (f00 + (f01 - f00) * fj) + (f10 - f00 + (f11 + f00 - f01 - f10) * fj) * fi
    got = np.array([wc.lamb(r, z, 1.0, 0.0, 6.0, 2) for r, z in p])
    if not np.allclose(got, ref, rtol=1e-9, atol=1e-12):
        pytest.skip("numba froze a different Mat_nu earlier in this process")


WEYL_POINTS = np.array(
    [
        [3.0, 0.05],
        [5.0, 1.0],
        [8.0, -0.3],
        [10.0, 4.0],
        [15.0, 0.1],
        [20.0, -12.0],
        [30.0, 30.0],
        [35.0, -0.05],
        [7.5, 0.2],  # rho > b, near (not on) the disk plane
        [12.0, -0.15],
        [9.0, 0.01],
    ]
)


@pytest.mark.parametrize("MD", WEYL_CASES)
def test_weyl_probe_matches_cpu(MD, tmp_path, monkeypatch):
    _gpu_or_skip()
    from bh_shadows.gpu import weyl_gpu as wg

    M, b, lam = _weyl_setup(tmp_path, monkeypatch, MD)
    _frozen_or_skip(lam)
    nu, lm, nr, nz = wg.probe_gpu(WEYL_POINTS, M, MD, b, lam)
    cpu = np.array(
        [
            [
                wc.nu(r, z, M, MD, b, 2),
                wc.lamb(r, z, M, MD, b, 2),
                wc.derNU(r, z, M, MD, b, 0, 2, 1e-6),
                wc.derNU(r, z, M, MD, b, 1, 2, 1e-6),
            ]
            for r, z in WEYL_POINTS
        ]
    )
    np.testing.assert_allclose(nu, cpu[:, 0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(lm, cpu[:, 1], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(nr, cpu[:, 2], rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(nz, cpu[:, 3], rtol=1e-3, atol=1e-5)


# |z| below the GPU stencil step (1e-3): a central difference would straddle the nuD kink at z=0.
WEYL_POINTS_NEAR_PLANE = np.array([[7.5, 5e-4], [12.0, -2e-4], [4.0, 3e-4], [20.0, -9e-4]])


def test_weyl_probe_near_disk_plane(tmp_path, monkeypatch):
    _gpu_or_skip()
    from bh_shadows.gpu import weyl_gpu as wg

    MD = 0.1
    M, b, lam = _weyl_setup(tmp_path, monkeypatch, MD)
    _frozen_or_skip(lam)
    pts = WEYL_POINTS_NEAR_PLANE
    nz = wg.probe_gpu(pts, M, MD, b, lam)[3]
    cpu = np.array([wc.derNU(r, z, M, MD, b, 1, 2, 1e-6) for r, z in pts])
    np.testing.assert_allclose(nz, cpu, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("MD", WEYL_CASES)
def test_weyl_geo_rhs_matches_cpu(MD, tmp_path, monkeypatch):
    _gpu_or_skip()
    from bh_shadows.gpu import weyl_gpu as wg

    M, b, lam = _weyl_setup(tmp_path, monkeypatch, MD)
    _frozen_or_skip(lam)
    rho0, z0, al, be = 15.0, 0.0, 0.1, 0.2
    states = np.array(
        [[r, 0.05 * (-1) ** i, z, 0.02 * (i % 3 - 1)] for i, (r, z) in enumerate(WEYL_POINTS)]
    )
    gpu = wg.geo_gpu(states, rho0, z0, M, MD, b, al, be, lam)
    cpu = np.array([wc.geo(0.0, s, M, al, be, rho0, z0, MD, b, 1e-6) for s in states])
    np.testing.assert_allclose(gpu, cpu, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("MD", WEYL_CASES)
def test_weyl_gpu_smoke(MD, tmp_path, monkeypatch):
    _gpu_or_skip()
    from bh_shadows.gpu import weyl_gpu as wg

    M, b, lam = _weyl_setup(tmp_path, monkeypatch, MD)
    rho0 = wc.solve_observer_rho(M, MD, b, z0=0.0, target_gpp_sqrt=15.0)
    h = np.deg2rad(20.0)
    alpha, beta = np.linspace(-h, 0, 8), np.linspace(h, 0, 8)
    Mat, Mz, st = wg.f_paral_gpu(rho0, 0.0, M, MD, b, alpha, beta, MD > 0, lam)
    assert Mat.shape == Mz.shape == st.shape == (8, 8)
    assert set(np.unique(st)) <= {0, 3}
    assert np.isfinite(Mat[st == 0]).all() and np.isfinite(Mz[st == 0]).all()


def test_weyl_rejects_placeholder_matrix():
    from bh_shadows.gpu import weyl_gpu as wg

    with pytest.raises(ValueError, match="not loaded"):
        wg.f_paral_gpu(10.0, 0.0, 1.0, 0.0, 6.0, [0.0], [0.0], False, np.zeros((4, 4)))


def test_weyl_device_recorded(tmp_path, monkeypatch, caplog):
    """`--device gpu` without a GPU runs on the CPU, warns and records device == cpu."""
    monkeypatch.setattr(gb, "gpu_available", lambda: False)
    _seed_matrix(tmp_path, monkeypatch, 1.0, 0.0, 6.0)
    cfg = sc.ShadowConfig.from_toml(
        "configs/schwarzschild_nodisk.toml",
        {"n": 16, "n_grid": WEYL_N_GRID, "device_opt": "gpu"},
    )
    with caplog.at_level("WARNING"):
        rs.trace_grid(cfg, progress=False)
    assert rs.RUN_STATS["device"] == "cpu"
    assert "using cpu" in caplog.text
    assert "gpu_cpu_fallback_pixels" not in rs.RUN_STATS


def test_weyl_gpu_fallback_retraces_status3(tmp_path, monkeypatch):
    """Status-3 pixels from the (stubbed) GPU kernel are re-traced on the CPU and counted."""
    from bh_shadows.gpu import weyl_gpu as wg

    _seed_matrix(tmp_path, monkeypatch, 1.0, 0.0, 6.0)
    base = {"n": 16, "n_grid": WEYL_N_GRID}
    cpu = rs.trace_grid(
        sc.ShadowConfig.from_toml(
            "configs/schwarzschild_nodisk.toml", base | {"device_opt": "cpu"}
        ),
        progress=False,
    )
    flagged = [(0, 1), (2, 3), (7, 7), (4, 0)]

    def stub(rho0, z0, M, MD, b, alpha, beta, disk_visible, lam):
        Mat, Mz = wc.f_paral(rho0, z0, M, MD, b, alpha, beta, rs.HDER, disk_visible)
        st = np.zeros(Mat.shape, dtype=np.int64)
        for i, j in flagged:
            if i < len(alpha):
                Mat[i, j], Mz[i, j], st[i, j] = -123.0, 456.0, 3
        return Mat, Mz, st

    monkeypatch.setattr(rs, "resolve_device", lambda opt, parallel=True: "gpu")
    monkeypatch.setattr(wg, "f_paral_gpu", stub)
    cfg = sc.ShadowConfig.from_toml(
        "configs/schwarzschild_nodisk.toml", base | {"device_opt": "gpu"}
    )
    gpu = rs.trace_grid(cfg, progress=False)
    assert rs.RUN_STATS["device"] == "gpu"
    assert rs.RUN_STATS["gpu_cpu_fallback_pixels"] == len(flagged)
    for k in cpu:
        np.testing.assert_allclose(gpu[k], cpu[k], rtol=0, atol=1e-8)  # serial vs prange rounding


def _near_status_change(status):
    """True where any pixel in the 3x3 neighbourhood has a different status."""
    pad = np.pad(status, 1, mode="edge")
    near = np.zeros(status.shape, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            near |= (
                pad[1 + di : 1 + di + status.shape[0], 1 + dj : 1 + dj + status.shape[1]] != status
            )
    return near


@pytest.mark.parametrize("cfg_file", ["schwarzschild_nodisk", "schwarzschild_disk"])
def test_weyl_gpu_agrees_with_cpu(cfg_file, tmp_path, monkeypatch):
    _gpu_or_skip()
    cfg0 = sc.ShadowConfig.from_toml(f"configs/{cfg_file}.toml")
    _seed_matrix(tmp_path, monkeypatch, cfg0.M, cfg0.MD, cfg0.b)
    wc.load_matrix(str(gm.ensure_matrix(cfg0.M, cfg0.MD, cfg0.b, WEYL_N_GRID)))
    _frozen_or_skip(wc.Mat_nu)
    base = {"n": 64, "name": "agree", "n_grid": WEYL_N_GRID}
    cpu = rs.trace_grid(
        sc.ShadowConfig.from_toml(f"configs/{cfg_file}.toml", base | {"device_opt": "cpu"}),
        progress=False,
    )
    gpu = rs.trace_grid(
        sc.ShadowConfig.from_toml(f"configs/{cfg_file}.toml", base | {"device_opt": "gpu"}),
        progress=False,
    )
    assert rs.RUN_STATS["device"] == "gpu"

    sc_, sg = cpu["status"], gpu["status"]
    mism = sc_ != sg
    assert mism.sum() <= 0.01 * sc_.size
    assert not (mism & ~_near_status_change(sc_)).any()
    both_escape = (sc_ == 0) & (sg == 0)
    for k in ("coord1", "coord2"):
        assert np.median(np.abs(cpu[k][both_escape] - gpu[k][both_escape])) < 1e-3
    both_disk = (sc_ == 2) & (sg == 2)
    # The seeded MD=0 table makes this a CPU-vs-GPU test, not a physics test (real table: H7).
    if cfg_file == "schwarzschild_disk":
        assert (sc_ == 2).sum() > 0
        assert both_disk.sum() >= 0.9 * (sc_ == 2).sum()
        assert np.median(np.abs(cpu["coord1"][both_disk] - gpu["coord1"][both_disk])) < 1e-2
    else:
        assert not (sc_ == 2).any()


def test_gpu_run_json_has_dispatch_stats(tmp_path, monkeypatch):
    _gpu_or_skip()
    import json

    monkeypatch.setattr(sc, "RUNS_DIR", tmp_path / "runs")
    cfg = sc.ShadowConfig.from_toml(
        "configs/kerr_a0.9_nodisk.toml", {"n": 16, "device_opt": "gpu", "name": "stats"}
    )
    out = rs.trace_grid(cfg, progress=False, chunk=4)
    rs.save_run(cfg, out, elapsed_s=1.0)
    meta = json.loads((cfg.run_dir / "run.json").read_text())
    assert meta["device"] == "gpu"
    assert meta["gpu_chunk_rows"] == 4
    assert meta["gpu_dispatch_s_max"] >= meta["gpu_dispatch_s_mean"] > 0
    assert {"retrace_s", "gpu_cpu_fallback_pixels"} <= meta.keys()


def _bench(*argv):
    import subprocess
    import sys

    return subprocess.run(  # noqa: S603
        [sys.executable, "scripts/gpu_bench.py", *argv], capture_output=True, text=True
    )


def test_gpu_bench_rejects_large_n():
    res = _bench("--config", "configs/kerr_a0.9_nodisk.toml", "--n", "128")
    assert res.returncode != 0
    assert "--n must be <= 64" in res.stderr


def test_weyl_gpu_agrees_real_quadrature_table(tmp_path):
    """Fresh process (own frozen `Mat_nu`) with a real quadrature table, n_grid=200."""
    import json

    _gpu_or_skip()
    res = _bench(
        *("--config", "configs/schwarzschild_disk.toml", "--n", "64", "--n-grid", "200"),
        *("--agree", "--matrices-dir", str(tmp_path)),
    )
    assert res.returncode == 0, res.stderr[-2000:]
    m = json.loads(res.stdout.strip().splitlines()[-1])
    assert m["status_mismatch_frac"] <= 0.01
    assert m["mismatch_off_boundary"] == 0
    assert m["cpu_disk_pixels"] > 0
    assert m["median_dcoord1_escape"] < 1e-3 and m["median_dcoord2_escape"] < 1e-3
    assert m["median_dcoord1_disk"] < 1e-2
