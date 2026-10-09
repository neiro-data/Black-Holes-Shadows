# SCHEMA.md

The on-disk data of this repository: the lambda lookup matrix, the run outputs and the `[plot]` keys.
Referenced from [README.md](../README.md).

The lambda lookup matrix is a compressed NumPy archive (`matrices/lambda_M*_MD*_b*_n*.npz`, key
`lam`). `np.savez_compressed` writes it, and `weyl_core.load_matrix` reads it. The runner writes one `outputs.npz` per run (keys `status`, `coord1`,
`coord2`, and `phi` for Kerr).

## Pipeline

```
bh-shadow config.toml
   |-- Weyl: ensure_matrix --> matrices/lambda_*.npz --> wc.load_matrix --> f_paral (or GPU)
   |-- Kerr: kc.f_paral (or GPU), no lookup matrix
   v
runs/<id>/outputs.npz + run.json + image.png
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

The file name records `M`, `MD`, `b` and `n_grid`. A JSON sidecar beside the file holds the same
parameters. `ensure_matrix` refuses a file whose sidecar does not match the request.

`load_matrix` sets a module-level matrix inside `weyl_core`. Call it before the first trace, or
the tracer runs against no matrix.

## Config key — [observer] off_plane

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `off_plane` | bool | `false` | Put the Weyl observer at `z = OBSERVER_Z_OFFSET` (a small constant in `shadow_config.py`) instead of `z = 0`. |

Only the Weyl backend with a disk (`MD > 0`) accepts `true`. Other runs raise `ValueError`. The
vertical (alpha) mirror symmetry is lost, so the runner traces all alpha rows and half of the beta
columns, then mirrors the columns. This doubles the ray count. With `false` the output does not change.

## Config keys — [plot]

All keys are optional. Visuals are a post-process of the saved arrays; they do not change tracing.

| key | values | default | notes |
| --- | --- | --- | --- |
| `palette` | colormap name | `"grey_black_white"` | used only when `background = "palette"` |
| `background` | `palette`, `black_hole`, `two_color`, `greyscale`, `four_color`, `image` | `"palette"` | `palette` is the legacy `plot_status` image |
| `grid` | bool | `false` | celestial-sphere lines; an error with `palette` and `black_hole` |
| `grid_step_deg` | float | `10.0` | 0 < x <= 90 |
| `grid_width_deg` | float | `0.4` | 0 < x < step/2 |
| `grid_mode` | str | `"theta"` | `theta`, `end_z` (Weyl only) or `flat_angle` (Weyl only: lines straight in flat space, bent by the black hole); needs `grid = true` |
| `disk_alpha` | float | `0.6` | 0 to 1; disk opacity for `two_color`, `four_color`, `image` |
| `light_source` | bool | `false` | white glow behind the black hole; only with `two_color`, `four_color` |
| `light_alpha` | float | `0.7` | 0 to 1; glow peak opacity |
| `light_radius_deg` | float | `10.0` | > 0; Gaussian sigma of the glow, in degrees |
| `shadow_edge` | bool | `false` | draw the analytic shadow edge (none for Weyl with `MD > 0`) |
| `image` | path | `""` | required for `background = "image"`; relative to the config file; equirectangular sky |
| `save`, `show` | bool | `true`, `false` | as before |

Colours: `four_color` is upper-left red, upper-right green, lower-left yellow, lower-right blue;
`two_color` is red (upper) over blue (lower); `greyscale` is a grey sky with a white disk;
`black_hole` is white sky and black shadow. Captured pixels are black, failed pixels red.

Directions come from the arrays already traced. Kerr: theta is `coord2`, phi is `phi`. Weyl:
theta is `pi - atan2(rho, z)` from `coord1`/`coord2`, and phi is a flat-sky approximation from the
pixel column (`pi + PHI_LEFT_SIGN * beta`), so Weyl grid phi lines and `image` columns are
approximate. A disk pixel has no direction behind the disk, so it is `disk_alpha * white` over a
fixed dark grey (`visuals.DISK_BACKDROP`, 0.25); `greyscale` draws it opaque white.

The light source is a Gaussian glow around the point behind the observer (`theta = pi - theta0`,
`phi = pi`; Weyl uses `theta0 = 90 deg`). Layers: sky, glow, grid, then captured/disk/failed.
