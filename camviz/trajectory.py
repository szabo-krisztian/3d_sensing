"""Route (OSRM road snapping), resampling and a kinematically plausible speed profile."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import requests

from .config import Setup

EARTH_R = 6378137.0


def fetch_osrm_route(waypoints: list[tuple[float, float]]) -> list[tuple[float, float]]:
    coords = ";".join(f"{lon},{lat}" for lat, lon in waypoints)
    url = f"https://router.project-osrm.org/route/v1/driving/{coords}?overview=full&geometries=geojson"
    r = requests.get(url, timeout=30, headers={"User-Agent": "camviz-sensor-visualiser/0.1"})
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM error: {data.get('message', data.get('code'))}")
    return [(lat, lon) for lon, lat in data["routes"][0]["geometry"]["coordinates"]]


def load_route(setup: Setup) -> list[tuple[float, float]]:
    t = setup.trajectory
    if not t.snap_to_roads:
        return t.waypoints
    cache = setup.path(t.route_cache) if t.route_cache else None
    if cache and cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("waypoints") == [list(w) for w in t.waypoints]:
            return [tuple(p) for p in cached["route"]]
    route = fetch_osrm_route(t.waypoints)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"waypoints": [list(w) for w in t.waypoints], "route": route}))
    return route


@dataclass
class Trajectory:
    lat: np.ndarray
    lon: np.ndarray
    s: np.ndarray          # arc length [m]
    bearing: np.ndarray    # rad, clockwise from north, unwrapped
    speed: np.ndarray      # m/s
    t: np.ndarray          # s

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def at(self, time: float) -> dict:
        time = min(max(time, 0.0), self.duration)
        return {
            "lat": float(np.interp(time, self.t, self.lat)),
            "lon": float(np.interp(time, self.t, self.lon)),
            "bearing": float(np.interp(time, self.t, self.bearing)),
            "speed": float(np.interp(time, self.t, self.speed)),
            "s": float(np.interp(time, self.t, self.s)),
        }


def _smooth(a: np.ndarray, n: int) -> np.ndarray:
    if n < 2:
        return a
    pad = np.pad(a, n, mode="edge")
    return np.convolve(pad, np.ones(n) / n, mode="same")[n:-n]


def build_trajectory(setup: Setup, step_m: float = 1.0) -> Trajectory:
    cfg = setup.trajectory
    route = np.asarray(load_route(setup))
    lat0 = math.radians(route[:, 0].mean())
    east = np.radians(route[:, 1]) * EARTH_R * math.cos(lat0)
    north = np.radians(route[:, 0]) * EARTH_R
    seg = np.hypot(np.diff(east), np.diff(north))
    keep = np.concatenate([[True], seg > 0.05])
    east, north = east[keep], north[keep]
    s_raw = np.concatenate([[0], np.cumsum(np.hypot(np.diff(east), np.diff(north)))])

    s = np.arange(0, s_raw[-1], step_m)
    e = _smooth(np.interp(s, s_raw, east), 7)
    n = _smooth(np.interp(s, s_raw, north), 7)
    e[0], n[0] = np.interp(0, s_raw, east), np.interp(0, s_raw, north)

    bearing = np.unwrap(np.arctan2(np.gradient(e), np.gradient(n)))
    bearing = _smooth(bearing, 9)
    curvature = np.abs(np.gradient(bearing, s))

    vmax = cfg.max_speed_kmh / 3.6
    v_lim = np.minimum(vmax, np.sqrt(cfg.max_lateral_accel / np.maximum(curvature, 1e-6)))
    v = v_lim.copy()
    v[0] = 0.0
    for i in range(1, len(v)):       # acceleration limit
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * cfg.max_accel * step_m))
    v[-1] = 0.0
    for i in range(len(v) - 2, -1, -1):  # braking limit
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * cfg.max_decel * step_m))

    v_avg = np.maximum((v[1:] + v[:-1]) / 2, 0.05)
    t = np.concatenate([[0], np.cumsum(step_m / v_avg)])

    lat = np.degrees(n / EARTH_R)
    lon = np.degrees(e / (EARTH_R * math.cos(lat0)))
    return Trajectory(lat, lon, s, bearing, v, t)
