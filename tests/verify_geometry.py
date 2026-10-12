"""Numerical sanity checks of the camera / map geometry.

Run: python3 tests/verify_geometry.py   (exits non-zero on failure)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camviz import lens, optics  # noqa: E402
from camviz.config import load_setup  # noqa: E402
from camviz.tiles import metres_per_px  # noqa: E402
from camviz.trajectory import build_trajectory  # noqa: E402

CONFIG = Path(__file__).resolve().parent.parent / "config" / "skoda_superb_setup.yaml"
failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def cam_dir(cam, az_deg, el_deg):
    """Vehicle-frame direction at azimuth (left +) / elevation (up +) in the camera frame."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    d_cam = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    return (cam.rotation @ d_cam)[None, :]


setup = load_setup(CONFIG)

for cam in setup.cameras:
    w, h = cam.resolution
    # 1. the stated HFOV lands exactly on the sensor's left/right edge
    u, _, _ = optics.project(cam, cam_dir(cam, -cam.hfov_deg / 2, 0))
    check(f"{cam.name}: HFOV/2 -> image edge", abs(u[0] - w / 2) < 0.5, f"u={u[0]:.1f}, W/2={w / 2}")
    # 2. just outside the HFOV is not visible, just inside is
    _, _, vin = optics.project(cam, cam_dir(cam, cam.hfov_deg / 2 - 0.2, 0))
    _, _, vout = optics.project(cam, cam_dir(cam, cam.hfov_deg / 2 + 0.2, 0))
    check(f"{cam.name}: visibility flips at the FOV edge", bool(vin[0]) and not vout[0])
    # 3. derived VFOV lands on the top edge
    _, v, _ = optics.project(cam, cam_dir(cam, 0, cam.vfov_deg / 2))
    check(f"{cam.name}: VFOV/2 -> image top", abs(v[0] - h / 2) < 0.5, f"v={v[0]:.1f}, H/2={h / 2}")
    # 4. focal length matches the datasheet EFL / pixel pitch
    if cam.efl_mm and cam.pixel_size_um:
        f_ds = cam.efl_mm * 1000 / cam.pixel_size_um
        check(f"{cam.name}: f_px == EFL / pixel", abs(cam.f_px - f_ds) < 1e-6, f"{cam.f_px:.0f} px")
    # 5. centre resolution: 0.1° off-axis moves the image by f * 0.1° (= px/deg * 0.1)
    u, _, _ = optics.project(cam, cam_dir(cam, -0.1, 0))
    check(f"{cam.name}: px/deg at centre", abs(u[0] / 0.1 - optics.px_per_deg(cam)) < 0.01 * optics.px_per_deg(cam),
          f"{u[0] / 0.1:.1f} vs {optics.px_per_deg(cam):.1f}")
    # 6. a pedestrian at the detection range really spans the threshold number of pixels
    #    (full projection of head and feet, pedestrian centred on the optical axis)
    r_det = optics.camera_ranges(setup, cam)[0]
    hh = setup.target_height_m / 2
    top = cam.rotation @ np.array([r_det, 0, hh])
    bot = cam.rotation @ np.array([r_det, 0, -hh])
    _, v, _ = optics.project(cam, np.stack([top, bot]))
    px = v[0] - v[1]
    check(f"{cam.name}: pedestrian at {r_det:.0f} m spans {setup.tiers[0].min_pixels:g} px",
          abs(px - setup.tiers[0].min_pixels) < 0.05 * setup.tiers[0].min_pixels, f"{px:.2f} px")

# 7. orientation conventions: yaw +90 looks left (+y), pitch < 0 looks down
for cam in setup.cameras:
    axis = cam.rotation @ np.array([1.0, 0, 0])
    exp = np.array([math.cos(math.radians(cam.yaw_deg)), math.sin(math.radians(cam.yaw_deg))])
    check(f"{cam.name}: optical axis yaw", np.allclose(axis[:2] / np.linalg.norm(axis[:2]), exp, atol=1e-6))
    check(f"{cam.name}: optical axis pitch",
          abs(math.degrees(math.asin(axis[2])) - cam.pitch_deg) < 1e-6)

# 8. lens model limits
check("lens: a=1 is rectilinear", abs(lens.g(0.5, 1.0) - math.tan(0.5)) < 1e-12)
check("lens: a=0 is equidistant", abs(lens.g(0.5, 0.0) - 0.5) < 1e-12)
check("lens: g_inv(g(x)) == x", all(abs(lens.g_inv(float(lens.g(0.7, a)), a) - 0.7) < 1e-9
                                    for a in (1.0, 0.5, 0.0, -0.4, -1.0)))

# 9. map scale (Web Mercator) and route length
check("map: z19 m/px at 47.5°N", abs(metres_per_px(47.5, 19) - 0.2016) < 0.001, f"{metres_per_px(47.5, 19):.4f}")
traj = build_trajectory(setup)
d = np.hypot(np.diff(traj.lat) * 111_195, np.diff(traj.lon) * 111_195 * np.cos(np.radians(traj.lat[:-1])))
check("route: arc length == haversine length", abs(traj.s[-1] - d.sum()) / d.sum() < 0.02,
      f"{traj.s[-1]:.0f} m vs {d.sum():.0f} m")
check("route: speed <= limit", traj.speed.max() <= setup.trajectory.max_speed_kmh / 3.6 + 1e-6,
      f"max {traj.speed.max() * 3.6:.1f} km/h")

print(f"\n{len(failures)} failure(s)")
sys.exit(1 if failures else 0)
