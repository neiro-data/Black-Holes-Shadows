"""Config for the end-to-end shadow runner: TOML loader, CLI overrides, validation, routing."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

HALF_ANGLE_DEFAULT = 0.5880026035  # arctan(10 / 15), the legacy field of view
RUNS_DIR = Path("runs")


@dataclass(frozen=True)
class ShadowConfig:
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
    # matrix (Weyl only)
    n_grid: int = 1200
    source_opt: str = "auto"
    # plot
    palette: str = "grey_black_white"
    save: bool = True
    show: bool = False

    def __post_init__(self) -> None:
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
        if self.source_opt not in ("auto", "analytic", "quadrature"):
            raise ValueError(f"matrix source must be auto|analytic|quadrature: {self.source_opt!r}")
        if self.disk_visible_opt not in ("auto", True, False):
            raise ValueError(
                f"disk_visible must be 'auto', true or false: {self.disk_visible_opt!r}"
            )
        if self.backend == "weyl" and self.a != 0:
            raise ValueError("backend 'weyl' requires a == 0")
        if self.backend == "kerr" and self.MD > 0:
            raise ValueError("backend 'kerr' requires MD == 0")
        if self.backend == "kerr" and self.disk_visible:
            raise ValueError("the Kerr backend has no disk (disk_visible must be false)")

    @property
    def backend(self) -> str:
        if self.backend_opt != "auto":
            return self.backend_opt
        return "kerr" if self.a != 0 else "weyl"

    @property
    def disk_visible(self) -> bool:
        if self.disk_visible_opt == "auto":
            return self.MD > 0
        return bool(self.disk_visible_opt)

    @property
    def matrix_source(self) -> str:
        if self.source_opt != "auto":
            return self.source_opt
        return "analytic" if self.MD == 0 else "quadrature"

    @property
    def run_name(self) -> str:
        return self.name or f"{self.backend}_M{self.M}_a{self.a}_MD{self.MD}_b{self.b}_n{self.n}"

    @property
    def run_dir(self) -> Path:
        return RUNS_DIR / self.run_name

    @classmethod
    def from_toml(cls, path: str | Path, overrides: dict | None = None) -> ShadowConfig:
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
        mx = raw.get("matrix", {})
        if "n_grid" in mx:
            flat["n_grid"] = mx["n_grid"]
        if "source" in mx:
            flat["source_opt"] = mx["source"]
        for key in ("palette", "save", "show"):
            if key in raw.get("plot", {}):
                flat[key] = raw["plot"][key]
        flat.update({k: v for k, v in (overrides or {}).items() if v is not None})
        return cls(**flat)

    def to_toml(self) -> str:
        d = asdict(self)
        disk = d["disk_visible_opt"]
        disk_s = f'"{disk}"' if isinstance(disk, str) else str(disk).lower()
        return (
            f'name = "{self.run_name}"\n\n'
            f"[spacetime]\nM = {d['M']}\na = {d['a']}\nMD = {d['MD']}\nb = {d['b']}\n\n"
            f"[observer]\ndistance = {d['distance']}\ntheta0 = {d['theta0']}\n\n"
            f"[grid]\nn = {d['n']}\nhalf_angle = {d['half_angle']}\n\n"
            f"[tracer]\ndisk_visible = {disk_s}\nparallel = {str(d['parallel']).lower()}\n"
            f'backend = "{d["backend_opt"]}"\n\n'
            f'[matrix]\nn_grid = {d["n_grid"]}\nsource = "{d["source_opt"]}"\n\n'
            f'[plot]\npalette = "{d["palette"]}"\nsave = {str(d["save"]).lower()}\n'
            f"show = {str(d['show']).lower()}\n"
        )

    def write_resolved(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "config.resolved.toml"
        path.write_text(self.to_toml())
        return path
