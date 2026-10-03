import json

import numpy as np
import pytest

import generate_matriz as gm
from shadow_config import ShadowConfig


def write_cfg(tmp_path, body):
    p = tmp_path / "c.toml"
    p.write_text(body)
    return p


def test_overrides_and_routing(tmp_path):
    p = write_cfg(tmp_path, "[spacetime]\nMD = 0.1\n")
    cfg = ShadowConfig.from_toml(p)
    assert cfg.backend == "weyl" and cfg.disk_visible and cfg.matrix_source == "quadrature"
    cfg = ShadowConfig.from_toml(p, {"MD": 0.0})
    assert not cfg.disk_visible and cfg.matrix_source == "analytic"
    cfg = ShadowConfig.from_toml(p, {"MD": 0.0, "a": 0.9})
    assert cfg.backend == "kerr"


@pytest.mark.parametrize(
    "ov",
    [{"a": 1.0}, {"a": -0.1}, {"MD": -1.0}, {"b": 0.0}, {"n": 7}, {"a": 0.5, "MD": 0.1}],
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


def test_timestamp_validation_and_run_dir(tmp_path):
    cfg_file = write_cfg(tmp_path, 'name = "x"\n')
    with pytest.raises(ValueError):
        ShadowConfig.from_toml(cfg_file, {"timestamp": "bad"})
    cfg = ShadowConfig.from_toml(cfg_file, {"timestamp": "20260101_000000"})
    assert cfg.run_dir.name == "20260101_000000_x"
