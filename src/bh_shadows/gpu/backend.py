"""Optional Apple GPU backend: GPU probe, device choice and kernel launch.

The module imports MLX only when it needs it, so it works on hosts without MLX. The GPU uses
float32 only. See docs/GPU.md.
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
    """Return True if the Apple GPU is available.

    Returns
    -------
    bool
        True on Apple Silicon with MLX installed and a GPU default device. The function does not
        raise.
    """
    try:
        if sys.platform != "darwin" or platform.machine() != "arm64":
            return False
        import mlx.core as mx

        return mx.default_device() == mx.gpu
    except Exception:
        return False


def resolve_device(opt: str, parallel: bool = True) -> str:
    """Return the device that runs the trace.

    Parameters
    ----------
    opt : {"auto", "cpu", "gpu"}
        Device option. "gpu" falls back to "cpu" (with a warning) if no GPU is available. "auto"
        selects "gpu" only if `AUTO_PREFERS_GPU` is set.
    parallel : bool, optional
        False for a serial run. "auto" then selects "cpu".

    Returns
    -------
    str
        "cpu" or "gpu".

    Raises
    ------
    ValueError
        If `opt` is not a known device.
    """
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
    """Run a Metal kernel with one thread for each item.

    The function compiles each kernel one time for each process and keeps it in a cache.

    Parameters
    ----------
    name : str
        Kernel name.
    header : str
        MSL code before the kernel body (structs and functions).
    source : str
        MSL kernel body.
    inputs : dict of str to np.ndarray
        Kernel inputs. Float arrays become float32. Integer arrays become int32.
    outputs : dict of str to np.dtype
        Output names and types (float32 or int32).
    n_threads : int
        Number of threads.
    out_len : int, optional
        Length of each output. The default is `n_threads`.

    Returns
    -------
    dict of str to np.ndarray
        Outputs as numpy arrays.
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
