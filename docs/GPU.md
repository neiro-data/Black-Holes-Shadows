# GPU backend

This document describes the GPU port of the two ray tracers (Kerr and Weyl). It tells you what
the port does, what the kernel headers contain, and where the GPU results differ from the CPU
results.

## Summary

- The GPU backend is optional. It runs only on Apple Silicon with MLX installed
  (`uv sync --extra gpu`).
- Each tracer has one Metal (MSL) kernel. Each kernel thread traces one pixel (one ray).
- The kernels use float32. Apple GPUs do not support float64.
- The CPU (numba, float64) stays the reference. The GPU path must agree with it (see the tests in
  `tests/test_gpu_backend.py`).

```bash
uv sync --extra gpu
uv run bh-shadow configs/kerr_a0.9_nodisk.toml --device gpu
uv run bh-shadow configs/schwarzschild_disk.toml --n 64 --device gpu
```

## Files

| File | Content |
| --- | --- |
| `src/bh_shadows/gpu/backend.py` | GPU probe (`gpu_available`), device choice (`resolve_device`), kernel launch and kernel cache (`launch`). |
| `src/bh_shadows/gpu/kerr_gpu.py` | Kerr kernel. Port of `core/kerr_core.py`. |
| `src/bh_shadows/gpu/weyl_gpu.py` | Weyl kernel (Schwarzschild with an optional Morgan-Morgan disk). Port of `core/weyl_core.py`. |
| `src/bh_shadows/runner/run_shadow.py` | Sends rows to the GPU in chunks. Re-traces failed pixels on the CPU. Writes the GPU statistics to `run.json`. |
| `scripts/gpu_bench.py` | Measures the time of one config. With `--agree`, compares CPU and GPU results. |

## What the port does

1. The runner selects the device. `resolve_device` reads `[tracer] device` or `--device`.
2. The runner splits the image rows into chunks of `GPU_CHUNK_ROWS` rows (32). It sends one kernel
   launch per chunk.
3. The host (float64) calculates the observer constants once. The kernel receives them in the
   `prm` input.
4. Each thread sets the initial state of its ray (`init_ray`). Then it integrates the geodesic
   with the RKF45 method (`rkf`) until the ray ends.
5. The kernel writes the end state and a status code for each pixel.
6. The runner finds the pixels with status 3 and traces them again on the CPU in float64
   (`retrace_points`).

The Weyl lambda table (`Mat_nu`) goes to the kernel as a 2-D float32 input (`lam`). The GPU does
not build the table. The CPU builds it (quadrature or analytic) and the GPU only reads it.

## Why the kernels are MSL strings

- A GPU cannot run Python. On Apple Silicon, GPU code must be in the Metal Shading Language
  (MSL). MSL is based on C++14, so it looks like C.
- `mx.fast.metal_kernel` compiles MSL source at run time. For this reason the kernels are Python
  strings (`_HEADER`, `_TRACE`) in the GPU modules.
- MLX array operations (numpy-style) are not used for the trace. Each ray has its own adaptive
  step, number of steps, retries and stop condition. With array operations, all rays must wait
  for the slowest ray at each step. A custom kernel gives one thread to each ray, and each thread
  stops when its ray ends.
- Numba GPU support is for NVIDIA (CUDA) only. Apple GPUs do not support float64, so the MSL code
  uses float32 (see "Differences from the CPU path").

## Kernel headers

`mx.fast.metal_kernel` takes two MSL strings: a header and a body. In each GPU module, `_HEADER`
is the header and `_TRACE` is the body.

**`_HEADER`** holds the shared device code:

| Item | Meaning |
| --- | --- |
| `TOL` | Error tolerance of one RKF45 step (`1e-4`, as on the CPU). |
| `HMAX` | Largest step size. The step is negative because the ray goes back in time. |
| `HDER_GPU` | (Weyl only) Relative step of the finite difference for the disk potential. |
| `struct P` | Parameters of one ray: mass, spin or disk values, observer position and the constant momenta `pt`, `pphi`. |
| metric functions (`gtt_`, `gpp_`, `nuSch_`, `nuD_`, ...) | Metric components and potentials. Each one is a float32 copy of the CPU function with the same name. |
| `C001` ... `C323` | (Kerr only) Christoffel symbols. |
| `lamb_` | (Weyl only) Bilinear look-up in the lambda table. It includes the clamp branch of the CPU version. |
| `geo` | Right-hand side of the geodesic equations. |
| `rkf` | One RKF45 attempt. It returns the 4th-order state and the error estimate. |
| `init_ray` | Initial state of the ray from the pixel angles `alpha`, `beta`. |

**`_TRACE`** is the kernel body. It holds the integration loop and the stop conditions.

The kernel inputs and outputs:

| Name | Meaning |
| --- | --- |
| `alpha`, `beta` | Pixel angles. Thread `idx` traces `(alpha[idx // nb], beta[idx % nb])`. |
| `prm` | Observer constants from the host. |
| `lam` | (Weyl only) Lambda table. |
| `out_r`, `out_th`, `out_phi` | (Kerr) End state of the ray. |
| `out_rho`, `out_z` | (Weyl) End state, with the same sentinels as `weyl_core.func`: escape `[rho, z]`, capture `[0.001, 0.001]`, disk hit `[rho, z + 50]`. |
| `out_st` | Status code. Kerr: 0 escaped, 1 captured, 3 failed. Weyl: 0 done, 3 failed. |

`_PROBE` and `_GEO` are debug kernels. The tests use them to compare the potentials and the
right-hand side with the CPU values.

## `run.json` keys from a GPU run

| Key | Meaning |
| --- | --- |
| `device` | Device that ran the trace (`cpu` or `gpu`). |
| `gpu_chunk_rows` | Rows in each kernel launch. |
| `gpu_dispatch_s_max`, `gpu_dispatch_s_mean` | Longest and mean wall time of one launch. |
| `gpu_cpu_fallback_pixels` | Number of status-3 pixels traced again on the CPU. |
| `retrace_s` | Time of the CPU re-trace. |
| `matrix_s` | (Weyl) Time to build or load the lambda table. |

## Differences from the CPU path

The float32 type causes these changes:

- **Step-size floor.** `HMIN_GPU = -1e-5`. Smaller steps do not change a float32 state.
- **Limits.** Each ray has a step limit (`MAX_STEPS`) and each step has a retry limit
  (`MAX_RETRIES`). A ray that reaches a limit gets status 3. The CPU has no limits.
  A Kerr step with error 0 grows by a fixed factor of 4. A NaN error gives status 3.
- **Non-finite state.** A NaN or Inf state gives status 3. The CPU continues or returns NaN.
- **Weyl derivatives.** The CPU uses a central difference with `h = 1e-6`. In float32 that step
  gives only noise. The GPU uses analytic derivatives for the Schwarzschild part. For the disk
  part it uses a central difference with the step `HDER_GPU * max(1, |x|)`. Near the disk plane
  (`0 < |z| < h`) it uses a one-sided difference, because `nuD` has a kink at `z = 0`.
- **Weyl `nuD` formula.** The GPU uses an equivalent form of `nuD` that prevents cancellation.
- **Weyl retry branch.** The CPU retry loop has legacy errors: it compares `h < hmin` on negative
  steps and it can exceed 500 retries. The GPU uses a clean floor and the retry limit.

## Limitations

- **Platform.** Apple Silicon and MLX only. On other hosts, `--device gpu` logs a warning and runs
  on the CPU.
- **`auto` device.** `auto` selects the CPU (`AUTO_PREFERS_GPU = False` in `gpu/backend.py`). Set
  it to `True` to make `auto` select the GPU.
- **Serial runs.** With `parallel = false`, the runner ignores `gpu` and runs on the CPU.
- **Precision.** The median difference from the CPU in the end coordinates is below `1e-3`.
  Pixels near the shadow edge can change class. Use the CPU for reference results.
- **Lambda table.** The GPU cannot build the table. The quadrature uses scalar adaptive
  `scipy.quad`, which has no GPU form.
- **Watchdog.** macOS can stop a GPU launch that runs too long. The limit is not documented. Keep
  each launch below about 1 s (see the chunk rule below).
- **Compile time.** The first launch in each process compiles the kernel (0.2 to 0.9 s). Later
  launches use the cache.
- **One config per process.** `weyl_core.Mat_nu` is fixed when numba compiles. Run each config in
  its own process.

## Measured results

Benchmark one config (n <= 64; one config per process). `--agree` compares CPU and GPU on the
same table.

```bash
uv run python scripts/gpu_bench.py --config configs/schwarzschild_disk.toml --n 64 --n-grid 1200 --device gpu --chunk 32
```

**Measured (Apple Silicon, 2026-10-04, n = 64, warm = median of 3, matrix cached):**

| Config | CPU warm | GPU warm (chunk 32) | Speedup | Dispatch max (chunk 32) |
| --- | --- | --- | --- | --- |
| `schwarzschild_disk` (n_grid 1200) | 1.82 s | 0.079 s | 23x | 0.032 s |
| `schwarzschild_nodisk` | 1.90 s | 0.063 s | 30x | 0.019 s |
| `kerr_a0.9_nodisk` | 0.81 s | 0.019 s | 43x | 0.019 s |

Cold start (first trace, includes numba/Metal compile): CPU 4-6 s, GPU 0.2-0.9 s. `matrix_s` is
0.05 s on a cached table; the quadrature build is separate and scales as `n_grid^2`.
No float32 fallback pixels in any of these runs. All speedups are above 1.5x, so
`AUTO_PREFERS_GPU` can be set to `True`.

**Chunk rule** (`T_BUDGET` = 1.0 s per dispatch; an assumption, the macOS watchdog limit is not
documented). Weyl at n = 64, disk: max dispatch 0.031 / 0.032 / 0.031 / 0.032 s for chunk
4 / 8 / 16 / 32, so it is launch-bound. Per-thread cost `c = 0.0318 / (32 x 32)` = 3.1e-5 s
(pessimistic), projected for n = 1024: `t(32) = c x 32 x 512` = 0.51 s <= 1 s. Kerr: real
n = 4096 run, 0.19 s mean per 16-row dispatch; max/mean ratio at n = 64 is q = 1.04, so
`t(32) = 2 x 0.19 x 1.04` = 0.40 s <= 1 s. Both backends get 32.

**Weyl CPU vs GPU agreement on the real quadrature table** (n = 64, n_grid = 1200, M = 0.9,
MD = 0.1): status mismatch 0 of 4096 pixels; median |d coord1| on escaped pixels 5.3e-4,
|d coord2| 6.3e-5; 1140 disk pixels on both devices, median |d coord1| 6.3e-4; 0 fallback pixels.
