"""One-parameter radial lens model.

Image radius r = f * g(theta, a), theta = angle from the optical axis:
    a = 1      pinhole (rectilinear)        r = f tan(theta)
    0 < a < 1  mild barrel distortion       r = f tan(a theta) / a
    a = 0      equidistant fisheye          r = f theta
    a < 0      stronger barrel (-> equisolid / orthographic-like)   r = f sin(|a| theta) / |a|

f is the focal length in pixels and fixes the resolution at the image centre;
`a` is fitted so that the stated HFOV lands exactly on the sensor edge.
"""
from __future__ import annotations

import math

import numpy as np

_EPS = 1e-9


def g(theta, a: float):
    if abs(a) < _EPS:
        return theta
    if a > 0:
        return np.tan(a * theta) / a
    return np.sin(-a * theta) / -a


def theta_max(a: float) -> float:
    """Largest angle for which g is defined and monotonic."""
    return math.inf if abs(a) < _EPS else math.pi / (2 * abs(a))


def g_inv(r: float, a: float) -> float:
    if abs(a) < _EPS:
        return r
    if a > 0:
        return math.atan(a * r) / a
    return math.asin(min(-a * r, 1.0)) / -a


def fit_shape(f_px: float, half_width_px: float, half_hfov_rad: float) -> tuple[float, str | None]:
    """Find a in [-1, 1] with f * g(half_hfov, a) == half_width. Returns (a, warning)."""
    target = half_width_px / f_px

    def val(a):
        return math.inf if half_hfov_rad >= theta_max(a) else float(g(half_hfov_rad, a))

    if target >= val(1.0):
        return 1.0, (f"HFOV narrower than a pinhole lens with this EFL allows "
                     f"(EFL implies {2 * math.degrees(math.atan(target)):.1f}°)")
    if target <= val(-1.0):
        return -1.0, "HFOV too wide for this EFL even with strong barrel distortion"
    lo, hi = -1.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if val(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, None
