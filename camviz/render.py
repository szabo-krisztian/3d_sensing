"""Frame layout and video rendering.

Plot coordinates of all ego-frame panels: X = right (= -y_vehicle), Y = forward (= x_vehicle).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch, Patch, Polygon, Rectangle, Wedge
from PIL import Image, ImageDraw

from . import optics
from .config import Setup
from .tiles import TileSource, latlon_to_px, metres_per_px
from .trajectory import EARTH_R, Trajectory, build_trajectory

BG = "#0d1117"
PANEL = "#161b22"
FG = "#e6edf3"
MUTED = "#8b949e"
GRID = "#30363d"
COVERAGE_COLORS = ["#e63946", "#f4a261", "#e9c46a", "#52b788", "#2d6a4f"]


def _style_axes(ax, title=None):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7)
    if title:
        ax.set_title(title, color=FG, fontsize=10.5, loc="left", pad=5, fontweight="bold")


def _cam_angle(cam) -> float:
    """Plot-frame angle (deg, CCW from +X) of the camera's optical axis."""
    return 90.0 + cam.yaw_deg


def _ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def draw_car_top(ax, v, zorder=10, lw=1.0):
    hw = v.width_m / 2
    ax.add_patch(FancyBboxPatch((-hw, v.x_rear), v.width_m, v.length_m,
                                boxstyle="round,pad=0,rounding_size=0.35",
                                fc="#dfe6ee", ec="#0b0f14", lw=lw, zorder=zorder))
    ws_y0, ws_y1 = v.wheelbase_m - 1.40, v.wheelbase_m - 0.62
    ax.add_patch(Polygon([(-hw + 0.12, ws_y0), (hw - 0.12, ws_y0), (hw - 0.2, ws_y1), (-hw + 0.2, ws_y1)],
                         fc="#2b3a4a", ec="none", zorder=zorder + 1))
    rw_y0, rw_y1 = v.x_rear + 0.35, v.x_rear + 0.80
    ax.add_patch(Polygon([(-hw + 0.2, rw_y0), (hw - 0.2, rw_y0), (hw - 0.14, rw_y1), (-hw + 0.14, rw_y1)],
                         fc="#2b3a4a", ec="none", zorder=zorder + 1))
    mw = (v.mirror_width_m - v.width_m) / 2
    for sgn in (-1, 1):
        x0 = sgn * hw if sgn > 0 else -hw - mw
        ax.add_patch(Rectangle((x0, v.mirror_x_m - 0.05), mw, 0.18, fc="#dfe6ee", ec="#0b0f14",
                               lw=lw * 0.6, zorder=zorder))


def draw_resolution_wedges(ax, setup: Setup, ranges, max_alpha=0.3, n_rings=28, zorder=4,
                           outline_fisheye=False, tier_arcs=True, lw=1.0):
    """FOV wedges out to the detection range, fading with pixel density (pixels on target)."""
    for ci, cam in enumerate(setup.cameras):
        cx, cy = -cam.position[1], cam.position[0]
        a, half = _cam_angle(cam), cam.hfov_deg / 2
        r_det = ranges[ci][0]
        fill = not (outline_fisheye and cam.projection == "fisheye")
        if fill:
            # pixels on target ~ 1/d: alpha follows log(px / detection threshold)
            edges = np.geomspace(max(r_det / 60, 0.3), r_det, n_rings + 1)
            edges[0] = 0.0
            for r0, r1 in zip(edges[:-1], edges[1:]):
                frac = math.log(r_det / max(r1, 1e-3)) / math.log(60)
                al = max_alpha * (0.15 + 0.85 * min(frac, 1.0))
                ax.add_patch(Wedge((cx, cy), r1, a - half, a + half, width=r1 - r0,
                                   fc=to_rgba(cam.color, al), ec="none", zorder=zorder))
        ax.add_patch(Wedge((cx, cy), r_det, a - half, a + half, fill=False,
                           ec=to_rgba(cam.color, 0.85), lw=lw, ls="-" if fill else "--", zorder=zorder + 1))
        if tier_arcs:
            for r in ranges[ci][1:]:
                ax.add_patch(Wedge((cx, cy), r, a - half, a + half, width=0, fill=False,
                                   ec=to_rgba(cam.color, 0.6), lw=lw * 0.7, ls=":", zorder=zorder + 1))


def pedestrian_mask() -> np.ndarray:
    """High-res pedestrian silhouette (H=350, W=130), float 0..1."""
    img = Image.new("L", (130, 350), 0)
    d = ImageDraw.Draw(img)
    d.ellipse((45, 0, 85, 44), fill=255)                          # head
    d.rounded_rectangle((30, 48, 100, 190), radius=18, fill=255)  # torso
    d.rounded_rectangle((8, 55, 30, 180), radius=10, fill=255)    # arms
    d.rounded_rectangle((100, 55, 122, 180), radius=10, fill=255)
    d.rounded_rectangle((34, 180, 63, 350), radius=10, fill=255)  # legs
    d.rounded_rectangle((67, 180, 96, 350), radius=10, fill=255)
    return np.asarray(img, dtype=float) / 255


class Renderer:
    def __init__(self, setup: Setup, verbose: bool = True):
        self.setup = setup
        self.r = setup.render
        self.verbose = verbose
        self.traj: Trajectory = build_trajectory(setup)
        self.ranges = [optics.camera_ranges(setup, c) for c in setup.cameras]
        m = self.r.get("map", {})
        self.max_zoom = int(m.get("max_zoom", 19))
        self.tiles = TileSource(m.get("tile_url", "https://tile.openstreetmap.org/{z}/{x}/{y}.png"),
                                setup.path(m.get("cache_dir", "data/tile_cache")),
                                m.get("user_agent", "camviz-sensor-visualiser/0.1"),
                                darken=float(m.get("darken", 0.55)),
                                desaturate=float(m.get("desaturate", 0.85)))
        self.attribution = m.get("attribution", "© OpenStreetMap contributors")
        self.ahead_kf = np.asarray(m.get("ahead_keyframes", [[0, 130]]), dtype=float)
        self.ahead_kf[:, 1] *= float(m.get("zoom_out", 1.0))
        self.behind_ratio = float(m.get("behind_ratio", 0.25))
        self.fill_alpha = float(m.get("fill_alpha", 0.3))

        lat0 = math.radians(float(np.mean(self.traj.lat)))
        self._east = np.radians(self.traj.lon) * EARTH_R * math.cos(lat0)
        self._north = np.radians(self.traj.lat) * EARTH_R
        self._coslat0 = math.cos(lat0)

        self.nf_height = float(self.r.get("near_field_height_m", 0.0))
        self.nf = optics.near_field_coverage(setup, float(self.r.get("near_field_m", 14)),
                                             height=self.nf_height)

    def log(self, *a):
        if self.verbose:
            print(*a, file=sys.stderr, flush=True)

    def groups(self):
        """Cameras grouped by `group`, preserving order: [(group, [cams])]."""
        out: dict[str, list] = {}
        for c in self.setup.cameras:
            out.setdefault(c.group, []).append(c)
        return list(out.items())

    def ahead_at(self, t_video: float) -> float:
        kf = self.ahead_kf
        if len(kf) == 1 or t_video <= kf[0, 0]:
            return float(kf[0, 1])
        for (t0, a0), (t1, a1) in zip(kf[:-1], kf[1:]):
            if t_video <= t1:
                return float(a0 + (a1 - a0) * _ease((t_video - t0) / (t1 - t0)))
        return float(kf[-1, 1])

    # ------------------------------------------------------------------ layout
    def build_figure(self):
        W, H = int(self.r.get("width_px", 1920)), int(self.r.get("height_px", 1080))
        dpi = 100
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
        self.fig, self.W, self.H, self.dpi = fig, W, H, dpi

        fig.text(0.008, 0.968, self.r.get("title", "Camera setup"), color=FG, fontsize=17,
                 fontweight="bold", va="center")
        fig.text(0.008, 0.940, f"{self.setup.vehicle.name}  ·  {len(self.setup.cameras)} cameras",
                 color=MUTED, fontsize=10, va="center")

        self._build_map(fig.add_axes([0.008, 0.015, 0.555, 0.905]))
        self._build_near(fig.add_axes([0.578, 0.475, 0.2, 0.445]))
        self._build_far(fig.add_axes([0.79, 0.475, 0.2, 0.445]))
        self._build_pixels(fig.add_axes([0.578, 0.015, 0.412, 0.39]))
        return fig

    def _build_map(self, ax):
        self.ax_map = ax
        _style_axes(ax)
        ax.set_xticks([]), ax.set_yticks([])
        bbox = ax.get_position()
        self.map_w = int(round(bbox.width * self.W))
        self.map_h = int(round(bbox.height * self.H))
        self.map_img = ax.imshow(np.zeros((self.map_h, self.map_w, 3), np.uint8),
                                 extent=(-1, 1, -1, 1), origin="upper", interpolation="bilinear", zorder=0)
        ax.set_aspect("equal")

        self.rings = []
        for r in (25, 50, 100, 200, 400):
            self.rings.append((r, ax.add_patch(plt.Circle((0, 0), r, fill=False, ec=(1, 1, 1, 0.18),
                                                          lw=0.8, ls="--", zorder=2)),
                               ax.text(0.7, r + 0.8, f"{r} m", color=(1, 1, 1, 0.5), fontsize=8, zorder=2)))

        self.route_line, = ax.plot([], [], color="#4cc9f0", lw=2.5, alpha=0.7, zorder=3,
                                   solid_capstyle="round")
        draw_resolution_wedges(ax, self.setup, self.ranges, max_alpha=self.fill_alpha,
                               outline_fisheye=False, tier_arcs=False)
        # ground-level blind zone around the car (same data as the near-field panel); the
        # wedges above are horizontal-only and ignore vertical FOV and body occlusion
        xs, ys, masks = self.nf
        v = self.setup.vehicle
        gx, gy = np.meshgrid(xs, ys)
        on_car = (gx >= v.x_rear) & (gx <= v.x_front) & (np.abs(gy) <= v.width_m / 2)
        blind = (masks.sum(0) == 0) & ~on_car
        rgba = np.zeros(blind.shape + (4,))
        rgba[blind] = to_rgba(COVERAGE_COLORS[0], 0.75)
        ax.imshow(rgba.transpose(1, 0, 2)[:, ::-1], origin="lower",
                  extent=(-ys[-1], -ys[0], xs[0], xs[-1]), interpolation="nearest", zorder=6)
        draw_car_top(ax, self.setup.vehicle, zorder=20, lw=0.6)
        self.car_marker, = ax.plot([0], [1.3], marker=(3, 0, 0), ms=11, color="white", mec="black",
                                   zorder=21, alpha=0)

        self.hud = ax.text(0.015, 0.985, "", transform=ax.transAxes, color=FG, fontsize=11,
                           va="top", family="monospace", zorder=30,
                           bbox=dict(fc=(0.05, 0.07, 0.09, 0.8), ec=GRID, boxstyle="round,pad=0.5"))
        ax.text(0.995, 0.005, self.attribution, transform=ax.transAxes, color=MUTED, fontsize=7,
                ha="right", va="bottom", zorder=30)

        handles = []
        for name, cams in self.groups():
            c = cams[0]
            n = f" ×{len(cams)}" if len(cams) > 1 else ""
            handles.append(Patch(fc=to_rgba(c.color, 0.5), ec=c.color,
                                 label=f"{name}{n}  ·  {c.hfov_deg:g}°"))
        leg = ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.85,
                        facecolor=PANEL, edgecolor=GRID, labelcolor=FG)
        leg.set_zorder(30)
        self._build_minimap(ax.inset_axes([0.73, 0.75, 0.26, 0.24]))

    def _build_minimap(self, ax):
        ax.set_zorder(25)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        lat, lon = self.traj.lat, self.traj.lon
        for z in range(18, 10, -1):
            xs, ys = zip(*(latlon_to_px(a, b, z) for a, b in zip(lat, lon)))
            if max(max(xs) - min(xs), max(ys) - min(ys)) < 420:
                break
        self.mini_z = z
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        span = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + 40
        img, ox, oy = self.tiles.mosaic(z, cx - span, cy - span, cx + span, cy + span)
        ax.imshow(np.asarray(img), extent=(ox, ox + img.width, oy + img.height, oy))
        ax.plot(xs, ys, color="#4cc9f0", lw=1.8)
        ax.set_xlim(cx - span, cx + span), ax.set_ylim(cy + span, cy - span)
        self.mini_dot, = ax.plot([], [], "o", ms=7, mfc="#ff4d6d", mec="white", mew=1.2)

    def _build_near(self, ax):
        v = self.setup.vehicle
        xs, ys, masks = self.nf
        _style_axes(ax, "Near-field coverage — " + ("ground level" if self.nf_height == 0
                                                     else f"{self.nf_height:g} m above ground"))
        count = masks.sum(0)
        rgba = np.array([to_rgba(COVERAGE_COLORS[min(c, len(COVERAGE_COLORS) - 1)], 0.85)
                         for c in range(count.max() + 1)])[count]
        ax.imshow(rgba.transpose(1, 0, 2)[:, ::-1], origin="lower",
                  extent=(-ys[-1], -ys[0], xs[0], xs[-1]), interpolation="nearest", zorder=1)
        for r in (5, 10):
            ax.add_patch(Rectangle((-v.width_m / 2 - r, v.x_rear - r), v.width_m + 2 * r,
                                   v.length_m + 2 * r, fill=False, ec=(1, 1, 1, 0.35), ls="--",
                                   lw=0.7, zorder=2))
            ax.text(v.width_m / 2 + r - 0.2, v.x_rear - r + 0.3, f"{r} m", color=FG, fontsize=7,
                    ha="right", zorder=2)
        draw_car_top(ax, v, zorder=10)
        for c in self.setup.cameras:
            ax.plot(-c.position[1], c.position[0], "o", ms=6, mfc=c.color, mec="black", mew=0.7,
                    zorder=15)
        ax.set_xlim(-ys[-1], -ys[0]), ax.set_ylim(xs[0], xs[-1])
        ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        n = min(len(COVERAGE_COLORS), count.max() + 1)
        labels = ["blind"] + [f"{i} cam{'s' if i > 1 else ''}" for i in range(1, n)]
        handles = [Patch(fc=COVERAGE_COLORS[i], label=labels[i]) for i in range(n)]
        ax.legend(handles=handles, loc="lower center", ncol=n, fontsize=7.5, framealpha=0.85,
                  facecolor=PANEL, edgecolor=GRID, labelcolor=FG, handlelength=1.0,
                  columnspacing=0.8, bbox_to_anchor=(0.5, -0.005))

    def _build_far(self, ax):
        far = float(self.r.get("far_field_m", 400))
        _style_axes(ax, "Range by resolution")
        draw_resolution_wedges(ax, self.setup, self.ranges, max_alpha=0.45, lw=0.9, zorder=3)
        for r in (50, 100, 200, 300, 400, 600):
            if r > far:
                break
            ax.add_patch(plt.Circle((0, 0), r, fill=False, ec=(1, 1, 1, 0.2), lw=0.6, ls="--", zorder=2))
            ax.text(r * 0.42 + 3, r * 0.906, f"{r} m", color=MUTED, fontsize=7, zorder=6)
        # detection-range labels on the forward / rearward cameras
        for ci, cam in enumerate(self.setup.cameras):
            r = self.ranges[ci][0]
            if r < 60 or (abs(cam.yaw_deg) > 1 and abs(abs(cam.yaw_deg) - 180) > 1):
                continue
            a = math.radians(_cam_angle(cam))
            rr = min(r, far * 0.96)
            ax.text(rr * math.cos(a), rr * math.sin(a), f"{r:.0f} m" + (" ↑" if r > far else ""),
                    color=cam.color, fontsize=8, fontweight="bold", zorder=8, ha="center", clip_on=True,
                    va="top" if r > far else ("bottom" if math.sin(a) > 0 else "top"))
        w = far * 0.6
        ax.set_xlim(-w, w), ax.set_ylim(-far * 0.5, far)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]), ax.set_yticks([])
        ax.plot(0, 0, marker=(3, 0, 0), ms=7, color="white", zorder=10)
        t = self.setup.tiers
        ax.text(0.97, 0.03, f"pedestrian ({self.setup.target_height_m:g} m) height in pixels\n"
                f"solid edge: {t[0].min_pixels:g} px ({t[0].name})" +
                "".join(f"\ndotted: {x.min_pixels:g} px ({x.name})" for x in t[1:]),
                transform=ax.transAxes, color=MUTED, fontsize=7, va="bottom", ha="right", zorder=9,
                bbox=dict(fc=PANEL, ec=GRID, alpha=0.85, boxstyle="round,pad=0.4"))

    def _build_pixels(self, ax):
        """'What a pedestrian looks like': pixelated silhouettes at their true pixel size."""
        _style_axes(ax, f"Pixel resolution — pedestrian ({self.setup.target_height_m:g} m) "
                        "as seen by each camera")
        ax.set_facecolor(BG)
        for s in ax.spines.values():
            s.set_visible(False)
        dists = [float(d) for d in self.r.get("pixel_distances_m", [10, 25, 50, 100, 200])]
        groups = self.groups()
        mask = pedestrian_mask()
        aspect = mask.shape[1] / mask.shape[0]
        th = self.setup.target_height_m
        label_w, col_w, row_h, fig_h = 2.2, 1.0, 1.0, 0.70

        for gi, (name, cams) in enumerate(groups):
            cam = cams[0]
            y0 = (len(groups) - 1 - gi) * row_h
            fy = optics.focal_px(cam)
            ax.text(0.05, y0 + 0.52, name, color=cam.color, fontsize=10.5, fontweight="bold", va="bottom")
            ax.text(0.05, y0 + 0.47, f"{cam.resolution[0]}×{cam.resolution[1]} · {cam.hfov_deg:g}°",
                    color=MUTED, fontsize=8, va="top")
            for di, d in enumerate(dists):
                px = fy * th / d
                n = max(1, int(round(px)))
                small = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize(
                    (max(1, int(round(n * aspect))), n), Image.BOX), dtype=float) / 255
                h_disp = 240
                big = np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize(
                    (int(h_disp * aspect), h_disp), Image.NEAREST), dtype=float) / 255
                rgba = np.zeros(big.shape + (4,))
                rgba[..., :3] = to_rgba(cam.color)[:3]
                rgba[..., 3] = big
                cx = label_w + di * col_w + col_w / 2
                hw = fig_h * aspect / 2 * 1.0
                ax.imshow(rgba, extent=(cx - hw, cx + hw, y0 + 0.2, y0 + 0.2 + fig_h),
                          interpolation="nearest", zorder=3)
                ax.text(cx, y0 + 0.16, f"{px:.0f} px", color=FG if round(px) >= self.setup.tiers[0].min_pixels
                        else COVERAGE_COLORS[0], fontsize=8.5, ha="center", va="top")
        for di, d in enumerate(dists):
            ax.text(label_w + di * col_w + col_w / 2, len(groups) * row_h + 0.02, f"{d:g} m",
                    color=MUTED, fontsize=9, ha="center", va="bottom", fontweight="bold")
        ax.set_xlim(0, label_w + len(dists) * col_w)
        ax.set_ylim(-0.05, len(groups) * row_h + 0.22)
        ax.set_xticks([]), ax.set_yticks([])

    # ------------------------------------------------------------------ frames
    def tile_zoom(self, lat: float, res: float) -> int:
        z = self.max_zoom
        while z > 1 and metres_per_px(lat, z) * 1.3 < res:
            z -= 1
        return z

    def update(self, t_sim: float, t_video: float):
        st = self.traj.at(t_sim)
        ahead = self.ahead_at(t_video)
        behind = ahead * self.behind_ratio
        res = (ahead + behind) / self.map_h
        side = self.map_w * res / 2
        ax = self.ax_map
        img = self.tiles.heading_up_view(self.tile_zoom(st["lat"], res), st["lat"], st["lon"],
                                         st["bearing"], self.map_w, self.map_h, res,
                                         (self.map_w / 2, ahead / res))
        self.map_img.set_data(img)
        self.map_img.set_extent((-side, side, -behind, ahead))
        ax.set_xlim(-side, side), ax.set_ylim(-behind, ahead)
        for r, circ, txt in self.rings:
            vis = r < ahead * 0.95 and r > ahead * 0.08
            circ.set_visible(vis), txt.set_visible(vis)
        self.car_marker.set_alpha(_ease((ahead - 150) / 60))

        e0 = math.radians(st["lon"]) * EARTH_R * self._coslat0
        n0 = math.radians(st["lat"]) * EARTH_R
        sel = (self.traj.s >= st["s"] - behind * 1.5) & (self.traj.s <= st["s"] + ahead * 2)
        de, dn = self._east[sel] - e0, self._north[sel] - n0
        cb, sb = math.cos(st["bearing"]), math.sin(st["bearing"])
        self.route_line.set_data(de * cb - dn * sb, de * sb + dn * cb)

        mx, my = latlon_to_px(st["lat"], st["lon"], self.mini_z)
        self.mini_dot.set_data([mx], [my])
        m, s = divmod(int(t_sim), 60)
        self.hud.set_text(f"speed  {st['speed'] * 3.6:5.1f} km/h\n"
                          f"dist   {st['s']:6.0f} m\n"
                          f"time   {m:02d}:{s:02d}  ({self.r.get('playback_speed', 1):g}x)")

    def frame_times(self):
        fps = float(self.r.get("fps", 30))
        speed = float(self.r.get("playback_speed", 1.0))
        n = int(self.traj.duration / speed * fps) + 1
        return [(i * speed / fps, i / fps) for i in range(n)], fps

    def prefetch(self, times):
        if not hasattr(self, "map_h"):
            self.build_figure()
        by_zoom: dict[int, float] = {}
        lat = float(np.mean(self.traj.lat))
        for _, tv in times[:: max(1, len(times) // 400)] + [times[-1]]:
            ahead = self.ahead_at(tv)
            res = ahead * (1 + self.behind_ratio) / self.map_h
            z = self.tile_zoom(lat, res)
            by_zoom[z] = max(by_zoom.get(z, 0), math.hypot(ahead, self.map_w * res / 2) + 30)
        for z, radius in sorted(by_zoom.items()):
            n_all, n_new = self.tiles.prefetch_route(
                z, self.traj.lat, self.traj.lon, radius,
                progress=lambda i, n: i % 50 == 0 and self.log(f"  downloading tiles {i}/{n}"))
            self.log(f"zoom {z}: {n_all} tiles, {n_new} downloaded")

    def preview(self, out: Path, t: float, t_video: float | None = None):
        self.build_figure()
        self.update(t, t_video if t_video is not None else 1e9)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(out, dpi=self.dpi, facecolor=BG)
        self.log(f"wrote {out}")

    def render_video(self, out: Path, max_seconds: float | None = None):
        times, fps = self.frame_times()
        if max_seconds:
            times = times[: int(max_seconds * fps)]
        self.log(f"route {self.traj.s[-1]:.0f} m, {self.traj.duration:.0f} s sim -> "
                 f"{len(times)} frames @ {fps:g} fps")
        self.build_figure()
        self.prefetch(times)
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=12000,
                              extra_args=["-pix_fmt", "yuv420p", "-preset", "medium"])
        # write to a temporary file so an interrupted render never clobbers the last good video
        tmp = out.with_name(out.stem + ".part" + out.suffix)
        with writer.saving(self.fig, str(tmp), self.dpi):
            for i, (ts, tv) in enumerate(times):
                self.update(ts, tv)
                writer.grab_frame(facecolor=BG)
                if i % 100 == 0:
                    self.log(f"frame {i}/{len(times)}")
        tmp.replace(out)
        self.log(f"wrote {out}")
