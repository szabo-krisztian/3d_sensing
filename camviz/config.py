"""Loading and validating the YAML sensor-setup config."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import lens


@dataclass
class Vehicle:
    name: str
    length_m: float
    width_m: float
    height_m: float
    wheelbase_m: float
    front_overhang_m: float
    mirror_width_m: float
    mirror_x_m: float

    @property
    def rear_overhang_m(self) -> float:
        return self.length_m - self.wheelbase_m - self.front_overhang_m

    @property
    def x_front(self) -> float:
        return self.wheelbase_m + self.front_overhang_m

    @property
    def x_rear(self) -> float:
        return -self.rear_overhang_m

    @property
    def x_center(self) -> float:
        return 0.5 * (self.x_front + self.x_rear)

    def body_boxes(self) -> list[np.ndarray]:
        """Coarse occlusion model as [[xmin,ymin,zmin],[xmax,ymax,zmax]] boxes:
        bumpers, body up to the shoulder line, a hood sloping down in steps, cabin."""
        hw = self.width_m / 2
        hood_start = self.wheelbase_m - 0.65          # windshield base
        boxes = [
            [[self.x_rear, -hw, 0.12], [self.x_front, hw, 0.72]],                     # bumpers / lower body
            [[self.x_rear + 0.25, -hw + 0.05, 0.72], [hood_start, hw - 0.05, 0.95]],  # trunk + beltline
            [[self.x_rear + 0.8, -hw + 0.2, 0.95], [self.wheelbase_m - 1.4, hw - 0.2, self.height_m]],
        ]
        # hood: 0.92 m at the windshield base down to 0.80 m near the front edge
        steps = [(0.92, 0.0, 0.40), (0.86, 0.40, 0.75), (0.80, 0.75, 1.0)]
        hood_len = self.x_front - 0.2 - hood_start
        for top, f0, f1 in steps:
            boxes.append([[hood_start + f0 * hood_len, -hw + 0.05, 0.72],
                          [hood_start + f1 * hood_len, hw - 0.05, top]])
        return [np.array(b) for b in boxes]


@dataclass
class Camera:
    name: str
    type: str
    model: str
    resolution: tuple[int, int]
    hfov_deg: float
    vfov_deg: float
    projection: str
    position: np.ndarray
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    color: str
    group: str = ""
    max_range_m: float | None = None
    efl_mm: float | None = None
    pixel_size_um: float | None = None
    vfov_spec_deg: float | None = None   # VFOV as written in the spec (vfov_deg is derived)
    f_px: float = 0.0                    # focal length in pixels (image centre)
    lens_a: float = 1.0                  # lens shape, see lens.py
    warnings: list = field(default_factory=list)

    @property
    def rotation(self) -> np.ndarray:
        """Camera -> vehicle rotation. Camera axes: x optical, y left, z up."""
        y, p, r = map(math.radians, (self.yaw_deg, -self.pitch_deg, self.roll_deg))
        rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
        rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
        return rz @ ry @ rx


@dataclass
class RangeTier:
    name: str
    min_pixels: float


@dataclass
class TrajectoryCfg:
    waypoints: list[tuple[float, float]]
    snap_to_roads: bool
    route_cache: Path | None
    max_speed_kmh: float
    max_lateral_accel: float
    max_accel: float
    max_decel: float


@dataclass
class Setup:
    vehicle: Vehicle
    cameras: list[Camera]
    target_height_m: float
    target_width_m: float
    tiers: list[RangeTier]
    trajectory: TrajectoryCfg
    render: dict = field(default_factory=dict)
    base_dir: Path = Path(".")

    def path(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.base_dir / p


def load_setup(path: str | Path) -> Setup:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    # Relative paths in the config are relative to the project root (parent of config/).
    base_dir = path.parent.parent if path.parent.name == "config" else path.parent

    vehicle = Vehicle(**raw["vehicle"])

    models = raw.get("camera_models", {})
    cameras = []
    for c in raw["cameras"]:
        if "camera_model" in c:
            if c["camera_model"] not in models:
                raise ValueError(f"{c['name']}: unknown camera_model {c['camera_model']!r}")
            c = {**models[c["camera_model"]], **c}
        proj = c.get("projection", "pinhole")
        if proj not in ("pinhole", "fisheye"):
            raise ValueError(f"{c['name']}: unknown projection {proj!r}")
        if proj == "pinhole" and c["hfov_deg"] >= 180:
            raise ValueError(f"{c['name']}: pinhole cannot have hfov >= 180, use 'fisheye'")
        w, h = c["resolution"]
        hfov = float(c["hfov_deg"])
        half = math.radians(hfov) / 2
        warnings = []
        efl, pix = c.get("efl_mm"), c.get("pixel_size_um")
        if efl and pix:
            f_px = efl * 1000.0 / pix
            a, warn = lens.fit_shape(f_px, w / 2, half)
            if warn:
                warnings.append(warn)
        elif proj == "pinhole":
            f_px, a = (w / 2) / math.tan(half), 1.0
        else:
            f_px, a = (w / 2) / half, 0.0
        vfov = 2 * math.degrees(lens.g_inv((h / 2) / f_px, a))
        vfov_spec = c.get("vfov_deg")
        if vfov_spec and abs(vfov_spec - vfov) > max(1.0, 0.05 * vfov):
            warnings.append(f"spec VFOV {vfov_spec:g}° is inconsistent with HFOV/resolution "
                            f"(square pixels give {vfov:.1f}°); using {vfov:.1f}°")
        cameras.append(Camera(
            name=c["name"],
            type=c.get("type", ""),
            model=c.get("model", ""),
            resolution=(int(w), int(h)),
            hfov_deg=hfov,
            vfov_deg=vfov,
            efl_mm=efl,
            pixel_size_um=pix,
            vfov_spec_deg=vfov_spec,
            f_px=f_px,
            lens_a=a,
            warnings=warnings,
            projection=proj,
            position=np.asarray(c["position_m"], dtype=float),
            yaw_deg=float(c.get("yaw_deg", 0)),
            pitch_deg=float(c.get("pitch_deg", 0)),
            roll_deg=float(c.get("roll_deg", 0)),
            color=c.get("color", "#ffffff"),
            group=c.get("group", c["name"]),
            max_range_m=c.get("max_range_m"),
        ))

    rc = raw.get("range_criteria", {})
    tiers = [RangeTier(t["name"], float(t["min_pixels"])) for t in rc.get("tiers", [])] or \
        [RangeTier("detection", 20)]

    t = raw["trajectory"]
    traj = TrajectoryCfg(
        waypoints=[tuple(map(float, w)) for w in t["waypoints"]],
        snap_to_roads=bool(t.get("snap_to_roads", True)),
        route_cache=Path(t["route_cache"]) if t.get("route_cache") else None,
        max_speed_kmh=float(t.get("max_speed_kmh", 50)),
        max_lateral_accel=float(t.get("max_lateral_accel", 2.0)),
        max_accel=float(t.get("max_accel", 1.5)),
        max_decel=float(t.get("max_decel", 2.0)),
    )

    return Setup(vehicle, cameras, float(rc.get("target_height_m", 1.5)),
                 float(rc.get("target_width_m", 0.5)), tiers, traj,
                 raw.get("render", {}), base_dir)
