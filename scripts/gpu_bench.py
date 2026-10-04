#!/usr/bin/env python
"""Time the tracer on one config (n <= 64) and print one JSON line per run.

    uv run python scripts/gpu_bench.py --config configs/schwarzschild_disk.toml --n 64 \
        --n-grid 1200 --device gpu --chunk 16 --repeats 3
    uv run python scripts/gpu_bench.py --config configs/schwarzschild_disk.toml --n 64 --agree

One config per process: `weyl_core.Mat_nu` is frozen at first numba compile. `--agree` traces on
the CPU, then on the GPU, in the same process (same frozen table) and prints the agreement.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import generate_matriz as gm  # noqa: E402
import run_shadow as rs  # noqa: E402
import shadow_config as sc  # noqa: E402

MAX_N = 64


def _cfg(args, device: str) -> sc.ShadowConfig:
    ov = {"n": args.n, "device_opt": device, "name": "bench"}
    if args.n_grid:
        ov["n_grid"] = args.n_grid
    return sc.ShadowConfig.from_toml(args.config, ov)


def _near_status_change(status: np.ndarray) -> np.ndarray:
    pad = np.pad(status, 1, mode="edge")
    h, w = status.shape
    near = np.zeros(status.shape, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            near |= pad[1 + di : 1 + di + h, 1 + dj : 1 + dj + w] != status
    return near


def bench(args) -> dict:
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
    cpu = rs.trace_grid(_cfg(args, "cpu"), progress=False)
    gpu = rs.trace_grid(_cfg(args, "gpu"), progress=False, chunk=args.chunk)
    if rs.RUN_STATS["device"] != "gpu":
        raise SystemExit("no GPU available for --agree")
    sc_, sg = cpu["status"], gpu["status"]
    mism = sc_ != sg
    both_escape = (sc_ == 0) & (sg == 0)
    both_disk = (sc_ == 2) & (sg == 2)

    def med(k, m):
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
