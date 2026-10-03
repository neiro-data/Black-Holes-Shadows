#!/usr/bin/env python
"""Config-driven end-to-end shadow runner: one config in, one traced + plotted shadow out.

    uv run python run_shadow.py configs/schwarzschild_disk.toml --MD 0.0 --b 6.0

One config per process: `weyl_core.Mat_nu` is frozen at first numba compile.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import time
from collections.abc import Callable

import numba
import numpy as np
from tqdm import tqdm

import weyl_core as wc
from generate_matriz import ensure_matrix
from gpu_backend import resolve_device
from shadow_config import ShadowConfig
from weyl_plot import classify_shadow_disk, mirror_quadrants, plot_status

log = logging.getLogger(__name__)

HDER = 10**-6
CORE_FRACTION = 0.8
RUN_STATS: dict = {}  # per-run device info from the last `trace_grid`, written to run.json
GPU_CHUNK_ROWS = 16  # rows per kernel launch; keeps each dispatch under the GPU watchdog


def limit_threads(fraction: float = CORE_FRACTION) -> int:
    """Cap numba threads at `fraction` of the system cores (at least 1); return the new count."""
    n = max(1, min(int((os.cpu_count() or 1) * fraction), numba.config.NUMBA_NUM_THREADS))
    numba.set_num_threads(n)
    return n


def _angles(cfg: ShadowConfig) -> tuple[np.ndarray, np.ndarray]:
    h = cfg.half_angle
    return np.linspace(-h, h, cfg.n), np.linspace(h, -h, cfg.n)


def _chunked_rows(
    fn: Callable[[int, int], tuple],
    n_rows: int,
    *,
    progress: bool,
    desc: str,
    chunk: int | None = None,
) -> tuple:
    """Call `fn(start, stop)` over row chunks, updating a progress bar; concat along axis 0."""
    chunk = max(1, chunk or numba.get_num_threads())
    parts = []
    with tqdm(total=n_rows, unit="row", desc=desc, disable=not progress) as pbar:
        for s in range(0, n_rows, chunk):
            e = min(s + chunk, n_rows)
            parts.append(fn(s, e))
            pbar.update(e - s)
    return tuple(np.concatenate(cols, axis=0) for cols in zip(*parts, strict=True))


def _mirror4(q: np.ndarray, sign_flip_lower: bool = False) -> np.ndarray:
    full = mirror_quadrants(q)
    if sign_flip_lower:
        full[q.shape[0] :, :] *= -1
    return full


def _trace_weyl(
    cfg: ShadowConfig, progress: bool = True, chunk: int | None = None
) -> dict[str, np.ndarray]:
    """Trace one quadrant (static, equatorially symmetric) and mirror both axes."""
    mpath = ensure_matrix(cfg.M, cfg.MD, cfg.b, cfg.n_grid, cfg.matrix_source)
    wc.load_matrix(str(mpath))  # must precede the first trace
    z0 = 0.0
    rho0 = wc.solve_observer_rho(cfg.M, cfg.MD, cfg.b, z0=z0, target_gpp_sqrt=cfg.distance)

    alfa_f, beta_f = _angles(cfg)
    half = cfg.n // 2
    alfa, beta = alfa_f[:half], beta_f[:half]
    if cfg.parallel:
        Mat, Mz = _chunked_rows(
            lambda s, e: wc.f_paral(
                rho0, z0, cfg.M, cfg.MD, cfg.b, alfa[s:e], beta, HDER, cfg.disk_visible
            ),
            half,
            progress=progress,
            desc="weyl",
            chunk=chunk,
        )
    else:
        Mat, Mz = np.zeros((half, half)), np.zeros((half, half))
        for i in tqdm(range(half), unit="row", desc="weyl", disable=not progress):
            for j in range(half):
                y = np.array(
                    [
                        rho0,
                        wc.dr(rho0, z0, cfg.M, cfg.MD, cfg.b, alfa[i], beta[j]),
                        z0,
                        wc.dthe(rho0, z0, cfg.M, cfg.MD, cfg.b, alfa[i]),
                    ]
                )
                Mat[i, j], Mz[i, j] = wc.func(
                    y,
                    300.0,
                    -0.02,
                    alfa[i],
                    beta[j],
                    cfg.M,
                    rho0,
                    z0,
                    cfg.MD,
                    cfg.b,
                    HDER,
                    cfg.disk_visible,
                )

    status = classify_shadow_disk(Mat, Mz, cfg.b)
    status[np.isnan(Mat) | np.isnan(Mz)] = 3
    return {
        "status": _mirror4(status).astype(int),
        "coord1": _mirror4(Mat),
        "coord2": _mirror4(Mz, sign_flip_lower=True),
    }


def _trace_kerr(
    cfg: ShadowConfig, progress: bool = True, chunk: int | None = None
) -> dict[str, np.ndarray]:
    """Trace the full beta range (frame dragging breaks left/right symmetry).

    Alpha is mirrored only at theta0 == 90 deg; otherwise the whole grid is traced.
    """
    import kerr_core as kc

    theta0 = np.deg2rad(cfg.theta0)
    alfa, beta = _angles(cfg)
    mirror = cfg.theta0 == 90.0
    alfa_t = alfa[: cfg.n // 2] if mirror else alfa

    device = resolve_device(cfg.device_opt, cfg.parallel)
    RUN_STATS["device"] = device
    if cfg.parallel:
        if device == "gpu":
            import kerr_gpu

            tracer = kerr_gpu.f_paral_gpu
        else:
            tracer = kc.f_paral
        Mr, Mth, Mphi, Mst = _chunked_rows(
            lambda s, e: tracer(cfg.distance, theta0, cfg.a, cfg.M, alfa_t[s:e], beta),
            len(alfa_t),
            progress=progress,
            desc="kerr-gpu" if device == "gpu" else "kerr",
            chunk=chunk if device == "cpu" else (chunk or GPU_CHUNK_ROWS),
        )
        if device == "gpu":
            RUN_STATS["gpu_cpu_fallback_pixels"] = 0
        if device == "gpu":  # float32 failures (status 3) are re-traced in float64 on the CPU
            for i, j in zip(*np.nonzero(Mst == 3), strict=True):
                y = np.array(
                    [
                        cfg.distance,
                        kc.dr(cfg.distance, cfg.a, cfg.M, theta0, alfa_t[i], beta[j]),
                        theta0,
                        kc.dthe(cfg.distance, cfg.a, cfg.M, theta0, alfa_t[i]),
                        0.0,
                    ]
                )
                Mr[i, j], Mth[i, j], Mphi[i, j], Mst[i, j] = kc.func_status(
                    y, 300.0, -0.02, alfa_t[i], beta[j], cfg.a, cfg.M, cfg.distance, theta0
                )
                RUN_STATS["gpu_cpu_fallback_pixels"] += 1
            log.info("gpu: %d pixels re-traced on cpu", RUN_STATS["gpu_cpu_fallback_pixels"])
    else:
        Mr, Mth, Mphi = (np.zeros((len(alfa_t), len(beta))) for _ in range(3))
        Mst = np.zeros((len(alfa_t), len(beta)), dtype=int)
        for i in tqdm(range(len(alfa_t)), unit="row", desc="kerr", disable=not progress):
            for j in range(len(beta)):
                y = np.array(
                    [
                        cfg.distance,
                        kc.dr(cfg.distance, cfg.a, cfg.M, theta0, alfa_t[i], beta[j]),
                        theta0,
                        kc.dthe(cfg.distance, cfg.a, cfg.M, theta0, alfa_t[i]),
                        0.0,
                    ]
                )
                Mr[i, j], Mth[i, j], Mphi[i, j], Mst[i, j] = kc.func_status(
                    y, 300.0, -0.02, alfa_t[i], beta[j], cfg.a, cfg.M, cfg.distance, theta0
                )

    out = {"status": Mst, "coord1": Mr, "coord2": Mth, "phi": Mphi}
    if mirror:  # equatorial reflection: theta -> pi - theta, r and phi unchanged
        out = {k: np.vstack([v, v[::-1]]) for k, v in out.items()}
        out["coord2"][len(alfa_t) :] = np.pi - out["coord2"][len(alfa_t) :]
    return out


def trace_grid(
    cfg: ShadowConfig, progress: bool = True, chunk: int | None = None
) -> dict[str, np.ndarray]:
    """Trace the full image for `cfg` and return status/coord1/coord2 (and phi for Kerr).

    `chunk` = rows per `f_paral` call in the parallel path (default: numba thread count).
    """
    RUN_STATS.clear()
    RUN_STATS["device"] = "cpu"  # the Weyl path has no GPU kernel yet
    trace = _trace_weyl if cfg.backend == "weyl" else _trace_kerr
    return trace(cfg, progress=progress, chunk=chunk)


def save_run(
    cfg: ShadowConfig,
    out: dict[str, np.ndarray],
    *,
    elapsed_s: float | None = None,
    config_path: str | None = None,
) -> None:
    run_dir = cfg.run_dir
    cfg.write_resolved()
    np.savetxt(run_dir / "status.txt", out["status"], fmt="%d")
    for key in ("coord1", "coord2", "phi"):
        if key in out:
            np.savetxt(run_dir / f"{key}.txt", out[key])
    if cfg.save or cfg.show:
        plot_status(out["status"], cfg, run_dir / "image.png" if cfg.save else None)
    meta = cfg.to_metadata() | {
        "config_path": config_path,
        "elapsed_s": elapsed_s,
        "shadow_pixels": int((out["status"] == 1).sum()),
        "device": RUN_STATS.get("device", "cpu"),
        **(
            {"gpu_cpu_fallback_pixels": RUN_STATS["gpu_cpu_fallback_pixels"]}
            if "gpu_cpu_fallback_pixels" in RUN_STATS
            else {}
        ),
        "numba_threads": numba.get_num_threads(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": numba.__version__,
        },
        "status": "completed",
    }
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")


def parse_args(argv=None) -> tuple[str, dict]:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--M", type=float)
    p.add_argument("--a", type=float)
    p.add_argument("--MD", type=float)
    p.add_argument("--b", type=float)
    p.add_argument("--n", type=int)
    p.add_argument("--name")
    p.add_argument("--disk-visible", choices=["auto", "true", "false"], dest="disk_visible_opt")
    p.add_argument("--device", choices=["auto", "cpu", "gpu"], dest="device_opt")
    p.add_argument("--no-progress", action="store_true", help="hide the progress bar")
    args = vars(p.parse_args(argv))
    cfg_path = args.pop("config")
    if args.pop("no_progress"):
        args["no_progress"] = True
    if args.get("disk_visible_opt") in ("true", "false"):
        args["disk_visible_opt"] = args["disk_visible_opt"] == "true"
    return cfg_path, {k: v for k, v in args.items() if v is not None}


def main(argv=None) -> None:
    cfg_path, overrides = parse_args(argv)
    no_progress = overrides.pop("no_progress", False)  # runner option, not a config field
    cfg = ShadowConfig.from_toml(cfg_path, overrides)
    threads = limit_threads()
    print(
        f"run_id={cfg.run_id} backend={cfg.backend} disk_visible={cfg.disk_visible} "
        f"run_dir={cfg.run_dir} threads={threads}"
    )
    t0 = time.time()
    out = trace_grid(cfg, progress=not no_progress)
    elapsed = time.time() - t0
    print(f"traced in {elapsed:.1f}s; shadow pixels: {int((out['status'] == 1).sum())}")
    save_run(cfg, out, elapsed_s=round(elapsed, 3), config_path=str(cfg_path))


if __name__ == "__main__":
    main()
