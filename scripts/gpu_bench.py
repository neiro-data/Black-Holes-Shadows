#!/usr/bin/env python
"""Measure the trace time of one config (n <= 64) and print one JSON line.

    uv run python scripts/gpu_bench.py --config configs/schwarzschild_disk.toml --n 64 \
        --n-grid 1200 --device gpu --chunk 16 --repeats 3
    uv run python scripts/gpu_bench.py --config configs/schwarzschild_disk.toml --n 64 --agree

Use one config for each process. Numba fixes `weyl_core.Mat_nu` at the first compile. `--agree`
traces on the CPU and then on the GPU in the same process (same table) and prints the agreement.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from bh_shadows.core import matrix as gm
from bh_shadows.runner import run_shadow as rs
from bh_shadows.runner import shadow_config as sc

MAX_N = 64


def _cfg(args, device: str) -> sc.ShadowConfig:
    """Load the config with the benchmark overrides.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments (`config`, `n`, `n_grid`).
    device : str
        Device option.

    Returns
    -------
    ShadowConfig
        Config named "bench".
    """
    ov = {"n": args.n, "device_opt": device, "name": "bench"}
    if args.n_grid:
        ov["n_grid"] = args.n_grid
    return sc.ShadowConfig.from_toml(args.config, ov)


def _near_status_change(status: np.ndarray) -> np.ndarray:
    """Return the pixels that touch a different status (8-neighbourhood).

    Parameters
    ----------
    status : np.ndarray
        Status image.

    Returns
    -------
    np.ndarray
        True where a neighbour has a different status.
    """
    pad = np.pad(status, 1, mode="edge")
    h, w = status.shape
    near = np.zeros(status.shape, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            near |= pad[1 + di : 1 + di + h, 1 + dj : 1 + dj + w] != status
    return near


def bench(args) -> dict:
    """Time one cold trace and `args.repeats` warm traces.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    dict
        Times (cold, median warm, matrix), dispatch statistics and fallback pixels.
    """
    cfg = _cfg(args, args.device)
    t0 = time.perf_counter()
    rs.trace_grid(cfg, progress=False, chunk=args.chunk)  # cold: matrix build + compiles
    cold = time.perf_counter() - t0
    matrix_cold = rs.RUN_STATS.get("matrix_s")
    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        rs.trace_grid(cfg, progress=False, chunk=args.chunk)
        times.append(time.perf_counter() - t0)
    st = rs.RUN_STATS
    return {
        "config": Path(args.config).stem,
        "n": args.n,
        "n_grid": cfg.n_grid,
        "device": st["device"],
        "chunk": args.chunk,
        "cold_s": round(cold, 3),
        "matrix_cold_s": matrix_cold,
        "warm_trace_s": round(statistics.median(times), 3),
        "warm_matrix_s": st.get("matrix_s"),
        "retrace_s": st.get("retrace_s"),
        "gpu_chunk_rows": st.get("gpu_chunk_rows"),
        "dispatch_s_max": st.get("gpu_dispatch_s_max"),
        "dispatch_s_mean": st.get("gpu_dispatch_s_mean"),
        "fallback_pixels": st.get("gpu_cpu_fallback_pixels"),
    }


def agree(args) -> dict:
    """Trace on the CPU, then on the GPU, and compare the results.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    dict
        Status mismatch, median coordinate differences and pixel counts.

    Raises
    ------
    SystemExit
        If no GPU is available.
    """
    cpu = rs.trace_grid(_cfg(args, "cpu"), progress=False)
    gpu = rs.trace_grid(_cfg(args, "gpu"), progress=False, chunk=args.chunk)
    if rs.RUN_STATS["device"] != "gpu":
        raise SystemExit("no GPU available for --agree")
    sc_, sg = cpu["status"], gpu["status"]
    mism = sc_ != sg
    both_escape = (sc_ == 0) & (sg == 0)
    both_disk = (sc_ == 2) & (sg == 2)

    def med(k, m):
        """Return the median |CPU - GPU| of key `k` on mask `m`, or None if `m` is empty."""
        return float(np.median(np.abs(cpu[k][m] - gpu[k][m]))) if m.any() else None

    return {
        "config": Path(args.config).stem,
        "n": args.n,
        "n_grid": _cfg(args, "cpu").n_grid,
        "status_mismatch_frac": float(mism.mean()),
        "mismatch_off_boundary": int((mism & ~_near_status_change(sc_)).sum()),
        "median_dcoord1_escape": med("coord1", both_escape),
        "median_dcoord2_escape": med("coord2", both_escape),
        "median_dcoord1_disk": med("coord1", both_disk),
        "cpu_disk_pixels": int((sc_ == 2).sum()),
        "both_disk_pixels": int(both_disk.sum()),
        "fallback_pixels": rs.RUN_STATS.get("gpu_cpu_fallback_pixels"),
    }


def main(argv=None) -> None:
    """Run the benchmark or the agreement check and print one JSON line.

    Parameters
    ----------
    argv : list of str, optional
        Arguments. The default is `sys.argv[1:]`.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--n", type=int, default=MAX_N)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--chunk", type=int)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--n-grid", type=int, dest="n_grid")
    p.add_argument("--matrices-dir", dest="matrices_dir")
    p.add_argument("--agree", action="store_true")
    args = p.parse_args(argv)
    if args.n > MAX_N:
        p.error(f"--n must be <= {MAX_N} (got {args.n})")
    if args.matrices_dir:
        gm.MATRICES_DIR = Path(args.matrices_dir)
    rs.limit_threads()
    print(json.dumps(agree(args) if args.agree else bench(args)))


if __name__ == "__main__":
    main()
