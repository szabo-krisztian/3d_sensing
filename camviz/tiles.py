"""Slippy-map tile fetching (disk + memory cache) and rotated, heading-up map views."""
from __future__ import annotations

import io
import math
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageEnhance

TILE = 256


def latlon_to_px(lat: float, lon: float, z: int) -> tuple[float, float]:
    """Global Web-Mercator pixel coordinates at zoom z."""
    n = TILE * 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def metres_per_px(lat: float, z: int) -> float:
    return 2 * math.pi * 6378137.0 * math.cos(math.radians(lat)) / (TILE * 2 ** z)


class TileSource:
    def __init__(self, url: str, cache_dir: Path, user_agent: str,
                 darken: float = 0.0, desaturate: float = 0.0):
        self.url = url
        self.cache_dir = Path(cache_dir)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.darken = darken
        self.desaturate = desaturate
        self._get = lru_cache(maxsize=2048)(self._load)

    def _fetch(self, z: int, x: int, y: int) -> bytes | None:
        path = self.cache_dir / str(z) / str(x) / f"{y}.png"
        if path.exists():
            return path.read_bytes()
        url = self.url.format(z=z, x=x, y=y, s="abc"[(x + y) % 3])
        for attempt in range(3):
            try:
                r = self.session.get(url, timeout=20)
                if r.status_code == 200:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(r.content)
                    return r.content
            except requests.RequestException:
                pass
            time.sleep(1 + attempt)
        return None

    def _load(self, z: int, x: int, y: int) -> Image.Image:
        n = 2 ** z
        data = self._fetch(z, x % n, y) if 0 <= y < n else None
        if data is None:
            img = Image.new("RGB", (TILE, TILE), (60, 60, 60))
        else:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        if self.desaturate:
            img = ImageEnhance.Color(img).enhance(1 - self.desaturate)
        if self.darken:
            img = ImageEnhance.Brightness(img).enhance(1 - self.darken)
        return img

    def tile(self, z: int, x: int, y: int) -> Image.Image:
        return self._get(z, x, y)

    def mosaic(self, z: int, x0: float, y0: float, x1: float, y1: float):
        """Image covering global pixel box [x0,x1]x[y0,y1]; returns (img, origin_x, origin_y)."""
        tx0, ty0 = int(math.floor(x0 / TILE)), int(math.floor(y0 / TILE))
        tx1, ty1 = int(math.floor(x1 / TILE)), int(math.floor(y1 / TILE))
        img = Image.new("RGB", ((tx1 - tx0 + 1) * TILE, (ty1 - ty0 + 1) * TILE))
        for tx in range(tx0, tx1 + 1):
            for ty in range(ty0, ty1 + 1):
                img.paste(self.tile(z, tx, ty), ((tx - tx0) * TILE, (ty - ty0) * TILE))
        return img, tx0 * TILE, ty0 * TILE

    def heading_up_view(self, z: int, lat: float, lon: float, bearing: float,
                        out_w: int, out_h: int, out_res: float, car_px: tuple[float, float]) -> np.ndarray:
        """Render a map view with the heading pointing up.

        bearing: rad clockwise from north. out_res: metres per output pixel.
        car_px: position of the vehicle in the output image (pixels).
        """
        px, py = latlon_to_px(lat, lon, z)
        k = out_res / metres_per_px(lat, z)
        cb, sb = math.cos(bearing), math.sin(bearing)
        cx, cy = car_px

        def to_in(i, j):
            return (px + k * ((i - cx) * cb + (cy - j) * sb),
                    py + k * ((i - cx) * sb - (cy - j) * cb))

        corners = [to_in(i, j) for i in (0, out_w) for j in (0, out_h)]
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        img, ox, oy = self.mosaic(z, min(xs) - 2, min(ys) - 2, max(xs) + 2, max(ys) + 2)
        a, b, c = k * cb, -k * sb, px - ox - k * cx * cb + k * cy * sb
        d, e, f = k * sb, k * cb, py - oy - k * cx * sb - k * cy * cb
        out = img.transform((out_w, out_h), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BILINEAR)
        return np.asarray(out)

    def prefetch_route(self, z: int, lats, lons, radius_m: float, progress=None):
        """Download all tiles within radius of the route up-front."""
        needed = set()
        for lat, lon in zip(lats[::5], lons[::5]):
            x, y = latlon_to_px(lat, lon, z)
            r = radius_m / metres_per_px(lat, z)
            for tx in range(int((x - r) // TILE), int((x + r) // TILE) + 1):
                for ty in range(int((y - r) // TILE), int((y + r) // TILE) + 1):
                    needed.add((tx, ty))
        missing = [t for t in sorted(needed)
                   if not (self.cache_dir / str(z) / str(t[0]) / f"{t[1]}.png").exists()]
        for i, (tx, ty) in enumerate(missing):
            self._fetch(z, tx, ty)
            if progress:
                progress(i + 1, len(missing))
        return len(needed), len(missing)
