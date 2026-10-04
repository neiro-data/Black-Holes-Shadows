"""Config for the end-to-end shadow runner: TOML loader, CLI overrides, validation, routing."""

from __future__ import annotations

import json
import math
import re
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from bh_shadows.plots.weyl_plot import PALETTES

BACKGROUNDS = ("palette", "black_hole", "two_color", "greyscale", "four_color", "image")
HALF_ANGLE_DEFAULT = 0.5880026035  # arctan(10 / 15), the legacy field of view
RUNS_DIR = Path("runs")
_STAMP_RE = re.compile(r"\d{8}_\d{6}")


def _now_stamp() -> str:
    """Return the local time as `YYYYMMDD_HHMMSS`."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class ShadowConfig:
    """Validated config of one run.

    The fields follow the TOML sections: spacetime (`M`, `a`, `MD`, `b`), observer (`distance`,
    `theta0`), grid (`n`, `half_angle`), tracer (`*_opt`, `parallel`), matrix (`n_grid`,
    `source_opt`) and plot. The `*_opt` fields hold the raw option. The properties give the resolved
    value. See docs/SCHEMA.md for each key.
    """

    name: str = ""
    # spacetime
    M: float = 1.0
    a: float = 0.0
    MD: float = 0.0
    b: float = 6.0
    # observer
    distance: float = 15.0
    theta0: float = 90.0
    # grid
    n: int = 160
    half_angle: float = HALF_ANGLE_DEFAULT
    # tracer
    disk_visible_opt: str | bool = "auto"
    parallel: bool = True
    backend_opt: str = "auto"
    device_opt: str = "auto"
    # matrix (Weyl only)
    n_grid: int = 1200
    source_opt: str = "auto"
    # plot
    palette: str = "grey_black_white"
    background: str = "palette"
    grid: bool = False
    grid_step_deg: float = 10.0
    grid_width_deg: float = 0.4
    disk_alpha: float = 0.6
    light_source: bool = False
    light_alpha: float = 0.7
    light_radius_deg: float = 10.0
    image_path: str = ""
    shadow_edge: bool = False
    save: bool = True
    show: bool = False
    # run identity (local time; not read from / written to TOML)
    timestamp: str = field(default_factory=_now_stamp)

    def __post_init__(self) -> None:
        """Validate the field values.

        Raises
        ------
        ValueError
            If a value is out of range or if two values do not agree (for example `M + MD != 1`, or
            `a != 0` with `MD > 0`).
        """
        if not _STAMP_RE.fullmatch(self.timestamp):
            raise ValueError(f"timestamp must be YYYYMMDD_HHMMSS, got {self.timestamp!r}")
        if self.palette not in PALETTES:
            raise ValueError(f"palette must be one of {sorted(PALETTES)}, got {self.palette!r}")
        if not math.isclose(self.M + self.MD, 1.0, abs_tol=1e-9):
            raise ValueError(
                f"M + MD must equal 1.0 (no disk: M = 1.0, MD = 0.0), got M={self.M}, MD={self.MD}"
            )
        if not 0 <= self.a < 1:
            raise ValueError(f"a must satisfy 0 <= a < 1, got {self.a}")
        if self.MD < 0:
            raise ValueError(f"MD must be >= 0, got {self.MD}")
        if self.b <= 0:
            raise ValueError(f"b must be > 0, got {self.b}")
        if self.n <= 0 or self.n % 2:
            raise ValueError(f"n must be a positive even integer, got {self.n}")
        if self.a != 0 and self.MD > 0:
            raise ValueError(
                "a != 0 with MD > 0 is not supported: the Weyl superposition only holds for "
                "static spacetimes (Kerr + disk is a separate plan)"
            )
        if self.backend_opt not in ("auto", "weyl", "kerr"):
            raise ValueError(f"backend must be auto|weyl|kerr, got {self.backend_opt!r}")
        if self.device_opt not in ("auto", "cpu", "gpu"):
            raise ValueError(f"device must be auto|cpu|gpu, got {self.device_opt!r}")
        if self.source_opt not in ("auto", "analytic", "quadrature"):
            raise ValueError(f"matrix source must be auto|analytic|quadrature: {self.source_opt!r}")
        if self.disk_visible_opt not in ("auto", True, False):
            raise ValueError(
                f"disk_visible must be 'auto', true or false: {self.disk_visible_opt!r}"
            )
        if self.background not in BACKGROUNDS:
            raise ValueError(f"background must be one of {BACKGROUNDS}, got {self.background!r}")
        if self.grid and self.background in ("palette", "black_hole"):
            raise ValueError(f"grid is not supported with background {self.background!r}")
        if not 0 < self.grid_step_deg <= 90:
            raise ValueError(f"grid_step_deg must satisfy 0 < x <= 90, got {self.grid_step_deg}")
        if not 0 < self.grid_width_deg < self.grid_step_deg / 2:
            raise ValueError(
                f"grid_width_deg must satisfy 0 < x < step/2, got {self.grid_width_deg}"
            )
        if not 0 <= self.disk_alpha <= 1:
            raise ValueError(f"disk_alpha must satisfy 0 <= x <= 1, got {self.disk_alpha}")
        if self.light_source and self.background not in ("two_color", "four_color"):
            raise ValueError(f"light_source is not supported with background {self.background!r}")
        if not 0 <= self.light_alpha <= 1:
            raise ValueError(f"light_alpha must satisfy 0 <= x <= 1, got {self.light_alpha}")
        if not self.light_radius_deg > 0:
            raise ValueError(f"light_radius_deg must be > 0, got {self.light_radius_deg}")
        if self.background == "image":
            if not self.image_path:
                raise ValueError("background 'image' requires plot.image")
            if not Path(self.image_path).is_file():
                raise ValueError(f"image not found: {self.image_path}")
        if self.backend == "weyl" and self.a != 0:
            raise ValueError("backend 'weyl' requires a == 0")
        if self.backend == "kerr" and self.MD > 0:
            raise ValueError("backend 'kerr' requires MD == 0")
        if self.backend == "kerr" and self.disk_visible:
            raise ValueError("the Kerr backend has no disk (disk_visible must be false)")

    @property
    def backend(self) -> str:
        """Tracer name: "weyl" or "kerr". "auto" selects "kerr" when `a != 0`."""
        if self.backend_opt != "auto":
            return self.backend_opt
        return "kerr" if self.a != 0 else "weyl"

    @property
    def disk_visible(self) -> bool:
        """True if rays stop at the disk. "auto" gives `MD > 0`."""
        if self.disk_visible_opt == "auto":
            return self.MD > 0
        return bool(self.disk_visible_opt)

    @property
    def matrix_source(self) -> str:
        """Lambda table method. "auto" gives "analytic" when `MD == 0`."""
        if self.source_opt != "auto":
            return self.source_opt
        return "analytic" if self.MD == 0 else "quadrature"

    @property
    def run_name(self) -> str:
        """Run name: `name`, or a name made from the parameters."""
        return self.name or f"{self.backend}_M{self.M}_a{self.a}_MD{self.MD}_b{self.b}_n{self.n}"

    @property
    def run_id(self) -> str:
        """Run identifier: `<timestamp>_<run_name>`."""
        return f"{self.timestamp}_{self.run_name}"

    @property
    def run_dir(self) -> Path:
        """Output folder: `RUNS_DIR / run_id`."""
        return RUNS_DIR / self.run_id

    def to_metadata(self) -> dict:
        """Return the run parameters for `run.json`.

        Returns
        -------
        dict
            Config values. The runner adds the run statistics.
        """
        started = datetime.strptime(self.timestamp, "%Y%m%d_%H%M%S").astimezone()
        weyl = self.backend == "weyl"
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "timestamp_iso": started.isoformat(),
            "name": self.run_name,
            "backend": self.backend,
            "M": self.M,
            "a": self.a,
            "MD": self.MD,
            "b": self.b,
            "distance": self.distance,
            "theta0": self.theta0,
            "n": self.n,
            "half_angle": self.half_angle,
            "disk_visible": self.disk_visible,
            "parallel": self.parallel,
            "device_opt": self.device_opt,
            "matrix_source": self.matrix_source if weyl else None,
            "n_grid": self.n_grid if weyl else None,
            "palette": self.palette,
            "background": self.background,
            "grid": self.grid,
            "image_path": self.image_path or None,
            "grid_step_deg": self.grid_step_deg,
            "grid_width_deg": self.grid_width_deg,
            "disk_alpha": self.disk_alpha,
            "light_source": self.light_source,
            "light_alpha": self.light_alpha,
            "light_radius_deg": self.light_radius_deg,
            "shadow_edge": self.shadow_edge,
        }

    @classmethod
    def from_toml(cls, path: str | Path, overrides: dict | None = None) -> ShadowConfig:
        """Load a config from a TOML file.

        Parameters
        ----------
        path : str or Path
            TOML file. A relative `[plot] image` path is resolved against the folder of this file.
        overrides : dict, optional
            Field values that replace the file values. None values are ignored.

        Returns
        -------
        ShadowConfig
            Validated config.
        """
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        flat = {"name": raw.get("name", "")}
        for key in ("M", "a", "MD", "b"):
            if key in raw.get("spacetime", {}):
                flat[key] = raw["spacetime"][key]
        for key in ("distance", "theta0"):
            if key in raw.get("observer", {}):
                flat[key] = raw["observer"][key]
        for key in ("n", "half_angle"):
            if key in raw.get("grid", {}):
                flat[key] = raw["grid"][key]
        tr = raw.get("tracer", {})
        if "disk_visible" in tr:
            flat["disk_visible_opt"] = tr["disk_visible"]
        if "parallel" in tr:
            flat["parallel"] = tr["parallel"]
        if "backend" in tr:
            flat["backend_opt"] = tr["backend"]
        if "device" in tr:
            flat["device_opt"] = tr["device"]
        mx = raw.get("matrix", {})
        if "n_grid" in mx:
            flat["n_grid"] = mx["n_grid"]
        if "source" in mx:
            flat["source_opt"] = mx["source"]
        plot = raw.get("plot", {})
        for key in (
            "palette",
            "background",
            "grid",
            "grid_step_deg",
            "grid_width_deg",
            "disk_alpha",
            "light_source",
            "light_alpha",
            "light_radius_deg",
            "shadow_edge",
            "save",
            "show",
        ):
            if key in plot:
                flat[key] = plot[key]
        if plot.get("image"):
            flat["image_path"] = str((Path(path).parent / plot["image"]).resolve())
        flat.update({k: v for k, v in (overrides or {}).items() if v is not None})
        return cls(**flat)

    def to_toml(self) -> str:
        """Return the config as TOML text.

        Returns
        -------
        str
            TOML text that `from_toml` can read again.
        """
        d = asdict(self)
        disk = d["disk_visible_opt"]
        disk_s = f'"{disk}"' if isinstance(disk, str) else str(disk).lower()
        return (
            f'name = "{self.run_name}"\n\n'
            f"[spacetime]\nM = {d['M']}\na = {d['a']}\nMD = {d['MD']}\nb = {d['b']}\n\n"
            f"[observer]\ndistance = {d['distance']}\ntheta0 = {d['theta0']}\n\n"
            f"[grid]\nn = {d['n']}\nhalf_angle = {d['half_angle']}\n\n"
            f"[tracer]\ndisk_visible = {disk_s}\nparallel = {str(d['parallel']).lower()}\n"
            f'backend = "{d["backend_opt"]}"\ndevice = "{d["device_opt"]}"\n\n'
            f'[matrix]\nn_grid = {d["n_grid"]}\nsource = "{d["source_opt"]}"\n\n'
            f'[plot]\npalette = "{d["palette"]}"\nbackground = "{d["background"]}"\n'
            f"grid = {str(d['grid']).lower()}\ngrid_step_deg = {d['grid_step_deg']}\n"
            f"grid_width_deg = {d['grid_width_deg']}\ndisk_alpha = {d['disk_alpha']}\n"
            f"light_source = {str(d['light_source']).lower()}\n"
            f"light_alpha = {d['light_alpha']}\nlight_radius_deg = {d['light_radius_deg']}\n"
            f"image = {json.dumps(d['image_path'])}\n"
            f"shadow_edge = {str(d['shadow_edge']).lower()}\n"
            f"save = {str(d['save']).lower()}\nshow = {str(d['show']).lower()}\n"
        )

    def write_resolved(self) -> Path:
        """Write the config to `run_dir / "config.resolved.toml"`.

        Returns
        -------
        Path
            Path of the written file.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "config.resolved.toml"
        path.write_text(self.to_toml())
        return path
