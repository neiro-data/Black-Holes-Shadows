# SCHEMA.md

The on-disk data of this repository: the lambda lookup matrix and the ray-tracing matrices.
Referenced from [README.md](README.md).

The lambda lookup matrix is a compressed NumPy archive (`matrices/lambda_M*_MD*_b*_n*.npz`, key
`lam`). `np.savez_compressed` writes it, and `weyl_core.load_matrix` reads it. `load_matrix` still
reads plain-text tables for the older scripts. The ray-tracing matrices are plain text:
`np.savetxt` writes them and `np.loadtxt` reads them.

## Pipeline

```
generate_matriz.py --> Mat_constA_Mbh_0.9        (the lambda lookup matrix)
                              |
                              v  wc.load_matrix + interpolation
   test_parallel_SHADOW.py / test_Z_SHADOW.py / test_symmetry_lensing.py
                              |
                              v  np.savetxt
                        Mat, Mz, Mphi
                              |
                              v
   symmetry.py / simetria_shadow*.py / lensing_image.py  --> figure

lensing_r_theta.py --> its own matrices --> figure       (Kerr, no lookup matrix)
```

## The lambda matrix

The lambda potential has a closed form only on the symmetry axis. Elsewhere the code must
integrate its derivatives numerically, which is too slow to repeat on every photon step. The
pipeline tabulates the potential once and interpolates it at run time.

| Property | Value |
| --- | --- |
| Shape | 1200 by 1200. |
| Row axis | `z`, from `40.0` down to `-40.0`. |
| Column axis | `rho`, from `40.0` down to `0.0`. |

The file name records the configuration of the run. Nothing parses the name. It is a label for a
person, and it is the reason a tracer can read the wrong file without an error. See
[README_AGENT.md — Matrix name against mode](README_AGENT.md#matrix-name-against-mode).

`load_matrix` sets a module-level matrix inside `weyl_core`. Call it before the first trace, or
the tracer runs against no matrix.

## The ray-tracing matrices

| File | Written by | Holds |
| --- | --- | --- |
| `Mat` | Every tracer | One value per pixel of the emission-angle grid. |
| `Mz` | The Weyl tracers | The `z` coordinate at the end of each ray. |
| `Mphi` | `test_symmetry_lensing.py` | The `phi` coordinate at the end of each ray. |

The emission-angle grids are `alfa` and `beta`. Each one spans `-arctan(10/15)` to
`arctan(10/15)`. The tracer then halves each grid and traces one quadrant. The plotting stage
mirrors that quadrant into the full image.

| Driver | Full grid | Traced quadrant |
| --- | --- | --- |
| `test_Z_SHADOW.py` | 80 by 80 | 40 by 40 |
| `lensing_r_theta.py` | 752 by 752 | 376 by 376 |
