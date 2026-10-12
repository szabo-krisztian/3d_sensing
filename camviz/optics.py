"""Camera models: projection, pixel density, resolution-derived ranges and ground coverage."""
from __future__ import annotations

import math

import numpy as np

from . import lens
from .config import Camera, Setup, Vehicle


def focal_px(cam: Camera) -> float:
    """Focal length in pixels: sets the resolution at the image centre."""
    return cam.f_px


def px_per_deg(cam: Camera) -> float:
    """Angular resolution at the image centre."""
    return cam.f_px * math.pi / 180


def tier_range(cam: Camera, target_height_m: float, min_pixels: float) -> float:
    """Distance at which a target of the given height spans `min_pixels` (image centre)."""
    r = cam.f_px * target_height_m / min_pixels
    return min(r, cam.max_range_m) if cam.max_range_m else r


def camera_ranges(setup: Setup, cam: Camera) -> list[float]:
    return [tier_range(cam, setup.target_height_m, t.min_pixels) for t in setup.tiers]


def project(cam: Camera, d: np.ndarray):
    """Project vehicle-frame direction vectors (N,3) to pixel offsets from the image centre.

    Returns (u, v, valid); u to the right, v up.
    """
    dc = d @ cam.rotation          # == (R^T d^T)^T: vehicle -> camera
    x_img, y_img = -dc[:, 1], dc[:, 2]
    w, h = cam.resolution
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.arccos(np.clip(dc[:, 0] / np.linalg.norm(dc, axis=1), -1, 1))
        ok = theta < lens.theta_max(cam.lens_a)
        r = np.where(ok, cam.f_px * lens.g(np.where(ok, theta, 0.0), cam.lens_a), np.inf)
        rho = np.hypot(x_img, y_img)
        rho = np.where(rho < 1e-12, 1.0, rho)
        u = r * x_img / rho
        v = r * y_img / rho
    valid = ok & (np.abs(u) <= w / 2) & (np.abs(v) <= h / 2)
    return u, v, valid


def _segment_hits_box(p0: np.ndarray, p1: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Slab test for segments p0 (3,) -> p1 (N,3) against an AABB. Ignores boxes containing p0."""
    if np.all(p0 >= box[0]) and np.all(p0 <= box[1]):
        return np.zeros(len(p1), dtype=bool)
    d = p1 - p0
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (box[0] - p0) / d
        t2 = (box[1] - p0) / d
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    return (tmax >= np.maximum(tmin, 1e-6)) & (tmin <= 1 - 1e-6)


def ground_visibility(cam: Camera, vehicle: Vehicle, pts: np.ndarray, max_range: float) -> np.ndarray:
    """Which ground points (N,3) the camera sees: inside the image, within range, not body-occluded."""
    d = pts - cam.position
    _, _, valid = project(cam, d)
    valid &= np.linalg.norm(d, axis=1) <= max_range
    idx = np.flatnonzero(valid)
    for box in vehicle.body_boxes():
        hit = _segment_hits_box(cam.position, pts[idx], box)
        idx = idx[~hit]
    out = np.zeros(len(pts), dtype=bool)
    out[idx] = True
    return out


def near_field_coverage(setup: Setup, half_size: float, res: float = 0.08, height: float = 0.0):
    """Per-camera visibility of points at `height` (0 = ground) on a grid centred on the vehicle.

    Returns (xs, ys, masks[n_cam, ny, nx]) in vehicle coordinates.
    """
    v = setup.vehicle
    xs = np.arange(v.x_center - half_size, v.x_center + half_size + res, res)
    ys = np.arange(-half_size, half_size + res, res)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, height)])
    masks = np.stack([
        ground_visibility(c, v, pts, camera_ranges(setup, c)[0]).reshape(gx.shape)
        for c in setup.cameras
    ])
    return xs, ys, masks


def blind_gap(vehicle: Vehicle, xs, ys, mask) -> float:
    """Closest visible ground point to the car body (m); inf if none."""
    if not mask.any():
        return math.inf
    gx, gy = np.meshgrid(xs, ys)
    dx = np.maximum(0, np.maximum(vehicle.x_rear - gx, gx - vehicle.x_front))
    dy = np.maximum(0, np.abs(gy) - vehicle.width_m / 2)
    return float(np.hypot(dx, dy)[mask].min())
