# SCHEMA.md

The on-disk data of this repository: the lambda lookup matrix, the run outputs and the `[plot]` keys.
Referenced from [README.md](../README.md).

The lambda lookup matrix is a compressed NumPy archive (`matrices/lambda_M*_MD*_b*_n*.npz`, key
`lam`). `np.savez_compressed` writes it, and `weyl_core.load_matrix` reads it. The runner writes one `outputs.npz` per run (keys `status`, `coord1`,
`coord2`, `phi`) for Kerr and Weyl.

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

## Config key — [observer] distance

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `distance` | float | `15.0` | Observer radius. Kerr: Boyer-Lindquist `r`. Weyl: circumferential radius `sqrt(g_phiphi)`. |

`distance` must be at most `ESCAPE_RADIUS` (30, in `shadow_config.py`). Every tracer marks a ray as
escaped at this radius, so an observer beyond it gets no shadow. A larger value raises
`ValueError`. At exactly 30, the Weyl tracer can mark every ray as escaped at the observer; Kerr
works. Keep well below 30 for image backgrounds: the end angles are read at the escape
radius, and the sky distorts when the observer is close to it.

## Config key — [observer] off_plane

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `off_plane` | bool | `false` | Put the Weyl observer above the plane instead of at `z = 0`. |
| `z0` | float | none | Observer height in Weyl units. Needs `off_plane = true`, Weyl, finite, `abs(z0) < distance`. Elevation is about `atan(z0 / distance)` (5 deg at distance 15 is `z0` about 1.3). |

The observer height (`ShadowConfig.observer_z`) is `z0` if set, else `OBSERVER_Z_OFFSET` (0.07, in
`shadow_config.py`) when `off_plane` is true, else 0. `run.json` stores `z0` and `observer_z`.
Orientation: for both backends `alpha > 0` looks up and `beta > 0` looks left. The Weyl tracer
negates the initial `dz/dlambda` because rays are traced backward, so the top image rows end at `z > 0`.

Only the Weyl backend (with or without a disk) accepts `true`. Kerr raises `ValueError`. The
vertical (alpha) mirror symmetry is lost, so the runner traces all alpha rows and half of the beta
columns, then mirrors the columns. This doubles the ray count. With `false` the output does not change.

## Config key — [tracer] disk_outer

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `disk_outer` | float | `inf` | Outer visible radius of the disk. A plane crossing at `rho >= disk_outer` is not a disk hit; the ray goes on. |

A visibility cut only: the metric (the Morgan-Morgan disk) is unchanged. Must be greater than `b`.
A finite value needs the Weyl backend. It has no effect with `disk_visible = false`. In `run.json`
infinity is stored as `null`. In `config.resolved.toml` it is written as `inf`.

## Config keys — [plot]

All keys are optional. Visuals are a post-process of the saved arrays; they do not change tracing.

| key | values | default | notes |
| --- | --- | --- | --- |
| `palette` | colormap name | `"grey_black_white"` | used only when `background = "palette"` |
| `background` | `palette`, `black_hole`, `two_color`, `greyscale`, `four_color`, `image`, `shaded_disk` | `"palette"` | `palette` is the legacy `plot_status` image |
| `grid` | bool | `false` | celestial-sphere lines; an error with `palette`, `black_hole` and `shaded_disk` |
| `grid_step_deg` | float | `10.0` | 0 < x <= 90 |
| `grid_width_deg` | float | `0.4` | 0 < x < step/2 |
| `grid_mode` | str | `"theta"` | `theta` or `flat_angle` (Weyl only: lines straight in flat space, bent by the black hole); needs `grid = true` |
| `disk_alpha` | float | `0.6` | 0 to 1; disk opacity for `two_color`, `four_color`, `image` |
| `light_source` | bool | `false` | white glow behind the black hole; only with `two_color`, `four_color` |
| `light_alpha` | float | `0.7` | 0 to 1; glow peak opacity |
| `light_radius_deg` | float | `10.0` | > 0; Gaussian sigma of the glow, in degrees |
| `shadow_edge` | bool | `false` | draw the analytic shadow edge (none for Weyl with `MD > 0`) |
| `image` | path | `""` | required for `background = "image"`; relative to the config file; equirectangular sky |
| `save`, `show` | bool | `true`, `false` | as before |

Colours: `four_color` is upper-left red, upper-right green, lower-left yellow, lower-right blue;
`two_color` is red (upper) over blue (lower); `greyscale` is a grey sky with a white disk;
`black_hole` is white sky and black shadow. `shaded_disk` (Weyl only; no grid) is a black sky and
shadow with only the disk drawn, white at `rho = b` and fading outward as `(b / rho)^0.75` (floor
0.15). Captured pixels are black, failed pixels red.

Directions come from the arrays already traced.
Old Weyl `outputs.npz` files store `coord2` with the opposite sign in plane mode (before the alpha
sign fix); re-rendered with this code their sky colours appear vertically flipped. Kerr: theta is `coord2`, phi is `phi`. Weyl:
theta is `atan2(rho, z)` from `coord1`/`coord2`, and phi is the traced `phi`. An old Weyl
`outputs.npz` without `phi` uses a flat-sky phi from the pixel column
(`pi + PHI_LEFT_SIGN * beta`) and logs a warning. A disk pixel has no direction behind the disk, so it is `disk_alpha * white` over a
fixed dark grey (`visuals.DISK_BACKDROP`, 0.25); `greyscale` draws it opaque white.

The light source is a Gaussian glow around the point behind the observer (`theta = pi - theta0`,
`phi = pi`; Weyl uses `theta0 = 90 deg`). Layers: sky, glow, grid, then captured/disk/failed.
