"""GPU backend: device resolution (no MLX needed) and CPU-vs-GPU agreement (skipped w/o MLX)."""

import numpy as np
import pytest

import gpu_backend as gb
import run_shadow as rs
import shadow_config as sc


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
    import kerr_core as kc
    import kerr_gpu as kg

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
