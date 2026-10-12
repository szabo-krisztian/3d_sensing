"""Loading and validating the YAML sensor-setup config."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


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
        bumpers, hood/trunk (shoulder line), cabin."""
        hw = self.width_m / 2
        lower = np.array([[self.x_rear, -hw, 0.12], [self.x_front, hw, 0.72]])
        shoulder = np.array([[self.x_rear + 0.25, -hw + 0.05, 0.72], [self.x_front - 0.35, hw - 0.05, 0.92]])
        cabin = np.array([[self.x_rear + 0.8, -hw + 0.2, 0.92],
                          [self.wheelbase_m - 1.4, hw - 0.2, self.height_m]])
        return [lower, shoulder, cabin]


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


def _derive_vfov(hfov: float, w: int, h: int, projection: str) -> float:
    if projection == "fisheye":
        return hfov * h / w
    return math.degrees(2 * math.atan(math.tan(math.radians(hfov) / 2) * h / w))


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
        cameras.append(Camera(
            name=c["name"],
            type=c.get("type", ""),
            model=c.get("model", ""),
            resolution=(int(w), int(h)),
            hfov_deg=float(c["hfov_deg"]),
            vfov_deg=float(c.get("vfov_deg") or _derive_vfov(c["hfov_deg"], w, h, proj)),
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
