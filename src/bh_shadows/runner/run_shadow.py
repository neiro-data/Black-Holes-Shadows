#!/usr/bin/env python
"""Run one shadow case from a TOML config: trace, save the arrays, and plot.

    uv run bh-shadow configs/schwarzschild_disk.toml --MD 0.0 --b 6.0

Use one config for each process. Numba fixes `weyl_core.Mat_nu` at the first compile.
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

from bh_shadows.core import weyl_core as wc
from bh_shadows.core.matrix import ensure_matrix, matrix_path
from bh_shadows.gpu.backend import resolve_device
from bh_shadows.plots import visuals
from bh_shadows.plots.weyl_plot import classify_shadow_disk, mirror_quadrants, plot_status
from bh_shadows.runner.shadow_config import ShadowConfig

log = logging.getLogger(__name__)

HDER = 10**-6
CORE_FRACTION = 0.8
RUN_STATS: dict = {}  # per-run device info from the last `trace_grid`, written to run.json
GPU_CHUNK_ROWS = 32  # rows per kernel launch; keeps each dispatch under the GPU watchdog (README)


def limit_threads(fraction: float = CORE_FRACTION) -> int:
    """Set the numba thread count to a fraction of the CPU cores.

    Parameters
    ----------
    fraction : float, optional
        Fraction of `os.cpu_count()`. The result is at least 1 and at most the numba limit.

    Returns
    -------
    int
        New thread count.
    """
    n = max(1, min(int((os.cpu_count() or 1) * fraction), numba.config.NUMBA_NUM_THREADS))
    numba.set_num_threads(n)
    return n


def _resolve_run_device(cfg: ShadowConfig) -> str:
    """Return the device for a run.

    The function always validates `cfg.device_opt`. A serial run (`parallel=False`) always uses the
    CPU.

    Parameters
    ----------
    cfg : ShadowConfig
        Run config.

    Returns
    -------
    str
        "cpu" or "gpu".
    """
    device = resolve_device(cfg.device_opt, cfg.parallel)
    if not cfg.parallel:
        if cfg.device_opt == "gpu":
            log.warning("device_opt='gpu' ignored: parallel=false runs serially on the cpu")
        return "cpu"
    return device


def _angles(cfg: ShadowConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return the pixel angle grids.

    Parameters
    ----------
    cfg : ShadowConfig
        Run config. Uses `n` and `half_angle`.

    Returns
    -------
    alpha, beta : np.ndarray
        `alpha` goes from -half_angle to half_angle. `beta` goes from half_angle to -half_angle.
    """
    h = cfg.half_angle
    return np.linspace(-h, h, cfg.n), np.linspace(h, -h, cfg.n)


def _chunked_rows(
    fn: Callable[[int, int], tuple],
    n_rows: int,
    *,
    progress: bool,
    desc: str,
    chunk: int | None = None,
    record_dispatch: bool = False,
) -> tuple:
    """Call a tracer on row chunks and join the results.

    Parameters
    ----------
    fn : callable
        `fn(start, stop)` traces rows `start:stop` and returns a tuple of arrays.
    n_rows : int
        Total number of rows.
    progress : bool
        Show a progress bar.
    desc : str
        Label of the progress bar.
    chunk : int, optional
        Rows in each call. The default is the numba thread count.
    record_dispatch : bool, optional
        Write the chunk size and the max and mean call times to `RUN_STATS`.

    Returns
    -------
    tuple of np.ndarray
        Each output of `fn`, joined along axis 0.
    """
    chunk = max(1, chunk or numba.get_num_threads())
    parts = []
    times = []
    with tqdm(total=n_rows, unit="row", desc=desc, disable=not progress) as pbar:
        for s in range(0, n_rows, chunk):
            e = min(s + chunk, n_rows)
            t0 = time.perf_counter()
            parts.append(fn(s, e))
            times.append(time.perf_counter() - t0)
            pbar.update(e - s)
    if record_dispatch and times:
        RUN_STATS["gpu_chunk_rows"] = chunk
        RUN_STATS["gpu_dispatch_s_max"] = round(max(times), 4)
        RUN_STATS["gpu_dispatch_s_mean"] = round(sum(times) / len(times), 4)
    return tuple(np.concatenate(cols, axis=0) for cols in zip(*parts, strict=True))


def _mirror4(
    q: np.ndarray,
    sign_flip_lower: bool = False,
    phi_flip_right: bool = False,
    phi0: float = 0.0,
) -> np.ndarray:
    """Mirror one quadrant into the full image.

    Parameters
    ----------
    q : np.ndarray
        Quadrant array.
    sign_flip_lower : bool, optional
        Change the sign of the lower half (use it for z).
    phi_flip_right : bool, optional
        Reflect phi about `phi0` on the right half (use it for phi: the beta mirror maps
        `phi - phi0` to `-(phi - phi0)`; the alpha mirror keeps phi).
    phi0 : float, optional
        Observer phi.

    Returns
    -------
    np.ndarray
        Array with twice the rows and twice the columns of `q`.
    """
    full = mirror_quadrants(q)
    if sign_flip_lower:
        full[q.shape[0] :, :] *= -1
    if phi_flip_right:
        full[:, q.shape[1] :] = phi0 - (full[:, q.shape[1] :] - phi0)
    return full


def _trace_weyl(
    cfg: ShadowConfig, progress: bool = True, chunk: int | None = None
) -> dict[str, np.ndarray]:
    """Trace a Weyl (static) run.

    The spacetime is symmetric about the equator and about the image centre. The function traces one
    quadrant and mirrors it. On the GPU, it traces the status-3 pixels again on the CPU.

    Parameters
    ----------
    cfg : ShadowConfig
        Run config.
    progress : bool, optional
        Show a progress bar.
    chunk : int, optional
        Rows in each tracer call.

    Returns
    -------
    dict of str to np.ndarray
        `status`, `coord1` (rho), `coord2` (z) and `phi`, each of shape (n, n).
    """
    log.info(
        "weyl run: lambda matrix %s",
        "found (cached)"
        if matrix_path(cfg.M, cfg.MD, cfg.b, cfg.n_grid).exists()
        else "not found, building now (may take a while for quadrature)",
    )
    t_mat = time.perf_counter()
    mpath = ensure_matrix(cfg.M, cfg.MD, cfg.b, cfg.n_grid, cfg.matrix_source)
    wc.load_matrix(str(mpath))  # must precede the first trace
    RUN_STATS["matrix_s"] = round(time.perf_counter() - t_mat, 3)
    z0 = 0.0
    rho0 = wc.solve_observer_rho(cfg.M, cfg.MD, cfg.b, z0=z0, target_gpp_sqrt=cfg.distance)

    alpha_full, beta_full = _angles(cfg)
    half = cfg.n // 2
    alpha, beta = alpha_full[:half], beta_full[:half]
    device = _resolve_run_device(cfg)
    RUN_STATS["device"] = device
    if cfg.parallel:
        if device == "gpu":
            from bh_shadows.gpu import weyl_gpu

            def tracer(s, e):
                """Trace rows `s:e` on the GPU."""
                return weyl_gpu.f_paral_gpu(
                    rho0, z0, cfg.M, cfg.MD, cfg.b, alpha[s:e], beta, cfg.disk_visible, wc.Mat_nu
                )
        else:

            def tracer(s, e):
                """Trace rows `s:e` on the CPU."""
                return wc.f_paral(
                    rho0, z0, cfg.M, cfg.MD, cfg.b, alpha[s:e], beta, HDER, cfg.disk_visible
                )

        Mat, Mz, Mphi, *gpu_st = _chunked_rows(
            tracer,
            half,
            progress=progress,
            desc="weyl-gpu" if device == "gpu" else "weyl",
            chunk=chunk if device == "cpu" else (chunk or GPU_CHUNK_ROWS),
            record_dispatch=device == "gpu",
        )
        if device == "gpu":  # float32 failures (status 3) are re-traced in float64 on the CPU
            fail = gpu_st[0] == 3
            RUN_STATS["gpu_cpu_fallback_pixels"] = int(fail.sum())
            t_re = time.perf_counter()
            if fail.any():
                ii, jj = np.nonzero(fail)
                Mat[ii, jj], Mz[ii, jj], Mphi[ii, jj] = wc.retrace_points(
                    rho0, z0, cfg.M, cfg.MD, cfg.b, alpha, beta, ii, jj, HDER, cfg.disk_visible
                )
            RUN_STATS["retrace_s"] = round(time.perf_counter() - t_re, 3)
            log.info("gpu: %d pixels re-traced on cpu", RUN_STATS["gpu_cpu_fallback_pixels"])
    else:
        Mat, Mz, Mphi = (np.zeros((half, half)) for _ in range(3))
        for i in tqdm(range(half), unit="row", desc="weyl", disable=not progress):
            for j in range(half):
                y = np.array(
                    [
                        rho0,
                        wc.dr(rho0, z0, cfg.M, cfg.MD, cfg.b, alpha[i], beta[j]),
                        z0,
                        wc.dthe(rho0, z0, cfg.M, cfg.MD, cfg.b, alpha[i]),
                        0.0,
                    ]
                )
                Mat[i, j], Mz[i, j], Mphi[i, j] = wc.func(
                    y,
                    300.0,
                    -0.02,
                    alpha[i],
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
        "phi": _mirror4(Mphi, phi_flip_right=True),
    }


def _trace_kerr(
    cfg: ShadowConfig, progress: bool = True, chunk: int | None = None
) -> dict[str, np.ndarray]:
    """Trace a Kerr run.

    Frame dragging breaks the left/right symmetry, so the function always traces all beta values. It
    mirrors alpha only when `theta0 == 90`. On the GPU, it traces the status-3 pixels again on the
    CPU.

    Parameters
    ----------
    cfg : ShadowConfig
        Run config.
    progress : bool, optional
        Show a progress bar.
    chunk : int, optional
        Rows in each tracer call.

    Returns
    -------
    dict of str to np.ndarray
        `status`, `coord1` (r), `coord2` (theta) and `phi`, each of shape (n, n).
    """
    from bh_shadows.core import kerr_core as kc

    theta0 = np.deg2rad(cfg.theta0)
    alpha, beta = _angles(cfg)
    mirror = cfg.theta0 == 90.0
    alpha_t = alpha[: cfg.n // 2] if mirror else alpha

    device = _resolve_run_device(cfg)
    RUN_STATS["device"] = device
    if cfg.parallel:
        if device == "gpu":
            from bh_shadows.gpu import kerr_gpu

            tracer = kerr_gpu.f_paral_gpu
        else:
            tracer = kc.f_paral
        Mr, Mth, Mphi, Mst = _chunked_rows(
            lambda s, e: tracer(cfg.distance, theta0, cfg.a, cfg.M, alpha_t[s:e], beta),
            len(alpha_t),
            progress=progress,
            desc="kerr-gpu" if device == "gpu" else "kerr",
            chunk=chunk if device == "cpu" else (chunk or GPU_CHUNK_ROWS),
            record_dispatch=device == "gpu",
        )
        if device == "gpu":  # float32 failures (status 3) are re-traced in float64 on the CPU
            t_re = time.perf_counter()
            ii, jj = np.nonzero(Mst == 3)
            if len(ii):
                Mr[ii, jj], Mth[ii, jj], Mphi[ii, jj], Mst[ii, jj] = kc.retrace_points(
                    cfg.distance, theta0, cfg.a, cfg.M, alpha_t, beta, ii, jj
                )
            RUN_STATS["retrace_s"] = round(time.perf_counter() - t_re, 3)
            RUN_STATS["gpu_cpu_fallback_pixels"] = len(ii)
            log.info("gpu: %d pixels re-traced on cpu", RUN_STATS["gpu_cpu_fallback_pixels"])
    else:
        Mr, Mth, Mphi = (np.zeros((len(alpha_t), len(beta))) for _ in range(3))
        Mst = np.zeros((len(alpha_t), len(beta)), dtype=int)
        for i in tqdm(range(len(alpha_t)), unit="row", desc="kerr", disable=not progress):
            for j in range(len(beta)):
                y = np.array(
                    [
                        cfg.distance,
                        kc.dr(cfg.distance, cfg.a, cfg.M, theta0, alpha_t[i], beta[j]),
                        theta0,
                        kc.dthe(cfg.distance, cfg.a, cfg.M, theta0, alpha_t[i]),
                        0.0,
                    ]
                )
                Mr[i, j], Mth[i, j], Mphi[i, j], Mst[i, j] = kc.func_status(
                    y, 300.0, -0.02, alpha_t[i], beta[j], cfg.a, cfg.M, cfg.distance, theta0
                )

    out = {"status": Mst, "coord1": Mr, "coord2": Mth, "phi": Mphi}
    if mirror:  # equatorial reflection: theta -> pi - theta, r and phi unchanged
        out = {k: np.vstack([v, v[::-1]]) for k, v in out.items()}
        out["coord2"][len(alpha_t) :] = np.pi - out["coord2"][len(alpha_t) :]
    return out


def trace_grid(
    cfg: ShadowConfig, progress: bool = True, chunk: int | None = None
) -> dict[str, np.ndarray]:
    """Trace the full image for a config.

    Parameters
    ----------
    cfg : ShadowConfig
        Run config. `cfg.backend` selects the Weyl or the Kerr tracer.
    progress : bool, optional
        Show a progress bar.
    chunk : int, optional
        Rows in each tracer call. The default is the numba thread count (CPU) or `GPU_CHUNK_ROWS`
        (GPU).

    Returns
    -------
    dict of str to np.ndarray
        `status`, `coord1`, `coord2`, and `phi` (Kerr only).
    """
    RUN_STATS.clear()
    trace = _trace_weyl if cfg.backend == "weyl" else _trace_kerr
    return trace(cfg, progress=progress, chunk=chunk)


def save_run(
    cfg: ShadowConfig,
    out: dict[str, np.ndarray],
    *,
    elapsed_s: float | None = None,
    config_path: str | None = None,
) -> None:
    """Write the run outputs to `cfg.run_dir`.

    The function writes `config.resolved.toml`, `outputs.npz` and `run.json`. It also writes
    `image.png` if `cfg.save` is set.

    Parameters
    ----------
    cfg : ShadowConfig
        Run config.
    out : dict of str to np.ndarray
        Result of `trace_grid`.
    elapsed_s : float, optional
        Trace time, written to `run.json`.
    config_path : str, optional
        Source config path, written to `run.json`.
    """
    run_dir = cfg.run_dir
    cfg.write_resolved()
    log.info("saving outputs.npz to %s", run_dir)
    t0 = time.perf_counter()
    np.savez_compressed(
        run_dir / "outputs.npz",
        **{k: out[k] for k in ("status", "coord1", "coord2", "phi") if k in out},
    )
    log.info("outputs.npz saved in %.1fs", time.perf_counter() - t0)
    if cfg.save or cfg.show:
        log.info("rendering image (background=%s, n=%d)", cfg.background, cfg.n)
        t0 = time.perf_counter()
        img_path = run_dir / "image.png" if cfg.save else None
        if cfg.background == "palette":
            plot_status(out["status"], cfg, img_path)
        else:
            visuals.render_image(out, cfg, img_path)
        log.info(
            "image done in %.1fs%s",
            time.perf_counter() - t0,
            f", saved {img_path}" if img_path else "",
        )
    meta = cfg.to_metadata() | {
        "config_path": config_path,
        "elapsed_s": elapsed_s,
        "shadow_pixels": int((out["status"] == 1).sum()),
        "device": RUN_STATS.get("device", "cpu"),
        **{
            k: RUN_STATS[k]
            for k in (
                "gpu_cpu_fallback_pixels",
                "gpu_chunk_rows",
                "gpu_dispatch_s_max",
                "gpu_dispatch_s_mean",
                "matrix_s",
                "retrace_s",
            )
            if k in RUN_STATS
        },
        "numba_threads": numba.get_num_threads(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": numba.__version__,
        },
        "outputs": "outputs.npz",
        "status": "completed",
    }
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("run complete: %s", run_dir)


def parse_args(argv=None) -> tuple[str, dict]:
    """Parse the command line of `bh-shadow`.

    Parameters
    ----------
    argv : list of str, optional
        Arguments. The default is `sys.argv[1:]`.

    Returns
    -------
    config_path : str
        Path of the TOML config.
    overrides : dict
        Config values from the command line. Unset options are not included.
    """
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
    """Run one config from the command line: trace, save and plot.

    Parameters
    ----------
    argv : list of str, optional
        Arguments. The default is `sys.argv[1:]`.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg_path, overrides = parse_args(argv)
    no_progress = overrides.pop("no_progress", False)  # runner option, not a config field
    cfg = ShadowConfig.from_toml(cfg_path, overrides)
    threads = limit_threads()
    print(
        f"run_id={cfg.run_id} backend={cfg.backend} disk_visible={cfg.disk_visible} "
        f"run_dir={cfg.run_dir} threads={threads}"
    )
    log.info("tracing %dx%d rays (backend=%s)", cfg.n, cfg.n, cfg.backend)
    t0 = time.time()
    out = trace_grid(cfg, progress=not no_progress)
    elapsed = time.time() - t0
    print(f"traced in {elapsed:.1f}s; shadow pixels: {int((out['status'] == 1).sum())}")
    save_run(cfg, out, elapsed_s=round(elapsed, 3), config_path=str(cfg_path))


if __name__ == "__main__":
    main()
