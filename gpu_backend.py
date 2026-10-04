"""Optional Apple-Silicon GPU backend: availability probe, device resolution, launch helper.

MLX is imported lazily so hosts without it (CI, Linux, Intel Macs) still import this module.
The GPU path is float32 only (Apple GPUs have no float64) and is opt-in.
"""

from __future__ import annotations

import logging
import platform
import sys

import numpy as np

log = logging.getLogger(__name__)

DEVICES = ("auto", "cpu", "gpu")
# Both spacetimes (Kerr, Weyl) have GPU kernels with CPU-vs-GPU agreement tests, but `auto` stays
# on the CPU until timing data justifies it; flipping this one flag makes `auto` prefer the GPU.
AUTO_PREFERS_GPU = False
THREADGROUP = 256

_kernel_cache: dict[tuple[str, str], object] = {}


def gpu_available() -> bool:
    """True on Apple Silicon with MLX importable and a GPU default device. Never raises."""
    try:
        if sys.platform != "darwin" or platform.machine() != "arm64":
            return False
        import mlx.core as mx

        return mx.default_device() == mx.gpu
    except Exception:
        return False


def resolve_device(opt: str, parallel: bool = True) -> str:
    """Map a `device` option (auto|cpu|gpu) to the device that will actually run."""
    if opt not in DEVICES:
        raise ValueError(f"device must be auto|cpu|gpu, got {opt!r}")
    if opt == "cpu":
        return "cpu"
    if opt == "gpu":
        if gpu_available():
            return "gpu"
        log.warning("device 'gpu' requested but no Apple GPU/MLX available; using cpu")
        return "cpu"
    return "gpu" if AUTO_PREFERS_GPU and parallel and gpu_available() else "cpu"


def launch(
    name: str,
    header: str,
    source: str,
    inputs: dict[str, np.ndarray],
    outputs: dict[str, np.dtype],
    n_threads: int,
    out_len: int | None = None,
) -> dict[str, np.ndarray]:
    """Run an MSL kernel (one thread per item) and return numpy outputs.

    `inputs` are cast to float32 (int arrays keep int32); the kernel is compiled once per
    process and cached by (name, source).
    """
    import mlx.core as mx

    key = (name, header + source)
    kernel = _kernel_cache.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=name,
            input_names=list(inputs),
            output_names=list(outputs),
            header=header,
            source=source,
        )
        _kernel_cache[key] = kernel
    args = [
        mx.array(
            np.ascontiguousarray(
                v, dtype=np.int32 if np.issubdtype(v.dtype, np.integer) else np.float32
            )
        )
        for v in inputs.values()
    ]
    dtypes = {"float32": mx.float32, "int32": mx.int32}
    res = kernel(
        inputs=args,
        grid=(n_threads, 1, 1),
        threadgroup=(min(THREADGROUP, max(1, n_threads)), 1, 1),
        output_shapes=[(out_len or n_threads,)] * len(outputs),
        output_dtypes=[dtypes[np.dtype(d).name] for d in outputs.values()],
    )
    mx.eval(*res)
    return {k: np.asarray(r) for k, r in zip(outputs, res, strict=True)}
