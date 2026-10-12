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


def draw_car_side(ax, v, zorder=10):
    xf, xr, wb, h = v.x_front, v.x_rear, v.wheelbase_m, v.height_m
    pts = [(xr + 0.05, 0.22), (xr, 0.55), (xr + 0.02, 0.95), (xr + 0.30, 1.02), (xr + 1.05, h - 0.02),
           (xr + 1.35, h), (wb - 1.55, h), (wb - 0.65, 0.98), (xf - 0.15, 0.80), (xf, 0.62),
           (xf - 0.05, 0.22)]
    ax.add_patch(Polygon(pts, fc="#dfe6ee", ec="#0b0f14", lw=0.8, zorder=zorder))
    for x in (0.0, wb):
        ax.add_patch(plt.Circle((x, 0.34), 0.34, fc="#1f2329", ec="#0b0f14", zorder=zorder + 1))


def draw_resolution_wedges(ax, setup: Setup, ranges, max_alpha=0.3, n_rings=30, zorder=4,
                           outline_fisheye=False, tier_arcs=True, lw=1.0):
    """FOV wedges out to the detection range, fading with pixel density (pixels on target).

    Full colour up to the strictest tier (e.g. identification), then fading with
    log(pixels on target) down to almost transparent at the detection range.
    """
    for ci, cam in enumerate(setup.cameras):
        cx, cy = -cam.position[1], cam.position[0]
        a, half = _cam_angle(cam), cam.hfov_deg / 2
        r_det, r_full = ranges[ci][0], ranges[ci][-1]
        fill = not (outline_fisheye and cam.projection == "fisheye")
        if fill:
            ax.add_patch(Wedge((cx, cy), r_full, a - half, a + half,
                               fc=to_rgba(cam.color, max_alpha), ec="none", zorder=zorder))
            edges = np.linspace(r_full, r_det, n_rings + 1)
            span = math.log(r_det / r_full) if r_det > r_full else 1.0
            for r0, r1 in zip(edges[:-1], edges[1:]):
                frac = math.log(r_det / ((r0 + r1) / 2)) / span   # 1 at r_full, 0 at r_det
                al = max_alpha * (0.06 + 0.94 * frac)
                ax.add_patch(Wedge((cx, cy), r1, a - half, a + half, width=r1 - r0,
                                   fc=to_rgba(cam.color, al), ec="none", zorder=zorder))
        ax.add_patch(Wedge((cx, cy), r_det, a - half, a + half, fill=False,
                           ec=to_rgba(cam.color, 0.5), lw=lw * 0.9, ls=(0, (5, 4)), zorder=zorder + 1))
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
        nf = self.r.get("near_field", {})
        v = setup.vehicle
        self.nf = optics.near_field_coverage(
            setup, float(nf.get("side_m", 8)), height=self.nf_height,
            x_range=(v.x_rear - float(nf.get("behind_m", 8)), v.x_front + float(nf.get("ahead_m", 17))))

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
        self._build_side(fig.add_axes([0.578, 0.715, 0.412, 0.19]))
        self._build_near(fig.add_axes([0.578, 0.015, 0.19, 0.645]))
        self._build_pixels(fig.add_axes([0.785, 0.015, 0.205, 0.645]))
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
        _style_axes(ax, "Near-field coverage\n" + ("at ground level" if self.nf_height == 0
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
        ax.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5, framealpha=0.85,
                  facecolor=PANEL, edgecolor=GRID, labelcolor=FG, handlelength=1.0,
                  columnspacing=0.8, bbox_to_anchor=(0.5, -0.005))

    def _build_side(self, ax):
        """Side view: vertical FOV of the forward / rearward cameras and the first visible ground point."""
        sv = self.r.get("side_view", {})
        x0, x1 = float(sv.get("x_min", -10)), float(sv.get("x_max", 30))
        v = self.setup.vehicle
        _style_axes(ax, "Side view — vertical field of view and ground blind distance")
        ax.axhline(0, color=MUTED, lw=1, zorder=1)
        seen_groups, n_lbl = set(), 0
        for ci, cam in enumerate(self.setup.cameras):
            cy = math.cos(math.radians(cam.yaw_deg))
            if abs(cy) < 0.9 or cam.group in seen_groups:
                continue
            seen_groups.add(cam.group)
            sgn = 1 if cy > 0 else -1
            px, pz = cam.position[0], cam.position[2]
            lim = x1 if sgn > 0 else x0
            half = min(cam.vfov_deg / 2, 89)

            def ray_end(el_deg):
                e = math.radians(el_deg)
                z = pz + abs(lim - px) * math.tan(e)
                if z >= 0:
                    return (lim, z), False
                return (px + sgn * pz / math.tan(-e), 0.0), True

            up, up_ground = ray_end(cam.pitch_deg + half)
            lo, lo_ground = ray_end(cam.pitch_deg - half)
            poly = [(px, pz), lo] + ([(lim, 0.0)] if lo_ground and not up_ground else []) + [up]
            ax.add_patch(Polygon(poly, closed=True, fc=to_rgba(cam.color, 0.16),
                                 ec=to_rgba(cam.color, 0.85), lw=1.0, zorder=3))
            ax.plot(px, pz, "o", ms=5, mfc=cam.color, mec="black", mew=0.6, zorder=12)
            # first visible ground point along the viewing direction (occlusion-aware)
            s_ = np.arange(0.0, 40.0, 0.05)
            yaw = math.radians(cam.yaw_deg)
            pts = np.column_stack([px + s_ * math.cos(yaw), cam.position[1] + s_ * math.sin(yaw),
                                   np.zeros_like(s_)])
            vis = optics.ground_visibility(cam, v, pts, self.ranges[ci][0])
            if vis.any():
                gx = pts[np.argmax(vis), 0]
                gap = max(gx - v.x_front if sgn > 0 else v.x_rear - gx, 0.0)
                ax.annotate("", xy=(gx, 0.05), xytext=(v.x_front if sgn > 0 else v.x_rear, 0.05),
                            arrowprops=dict(arrowstyle="<->", color=cam.color, lw=1.2), zorder=7)
                ax.text(gx + sgn * 0.3, -0.35 - 0.75 * (n_lbl % 2), f"{cam.group}: {gap:.1f} m",
                        color=cam.color, fontsize=8, ha="left" if sgn > 0 else "right", va="top",
                        zorder=7, fontweight="bold")
                n_lbl += 1
        draw_car_side(ax, v, zorder=10)
        ax.set_xlim(x0, x1), ax.set_ylim(-2.0, 6.0)
        ax.set_aspect("equal", adjustable="box", anchor="N")
        ax.set_yticks([0, 2, 4])
        ax.set_xticks(range(int(x0), int(x1) + 1, 5))
        ax.set_xticklabels([f"{t} m" for t in range(int(x0), int(x1) + 1, 5)])

    def _build_pixels(self, ax):
        """'What a pedestrian looks like': pixelated silhouettes at their true pixel size.
        One column per camera group, one row per distance."""
        _style_axes(ax, f"Pixel resolution\npedestrian ({self.setup.target_height_m:g} m) per camera")
        ax.set_facecolor(BG)
        for s in ax.spines.values():
            s.set_visible(False)
        dists = [float(d) for d in self.r.get("pixel_distances_m", [10, 25, 50, 100, 200])]
        groups = self.groups()
        mask = pedestrian_mask()
        aspect = mask.shape[1] / mask.shape[0]
        th = self.setup.target_height_m
        label_w, col_w, row_h, head_h, fig_h = 0.8, 1.0, 1.0, 0.9, 0.62
        xspan, yspan = label_w + len(groups) * col_w, head_h + len(dists) * row_h
        bbox = ax.get_position()
        ux, uy = bbox.width * self.W / xspan, bbox.height * self.H / yspan   # px per data unit
        hw = fig_h * uy * aspect / ux / 2

        for gi, (name, cams) in enumerate(groups):
            cam = cams[0]
            cx = label_w + gi * col_w + col_w / 2
            top = len(dists) * row_h
            ax.text(cx, top + 0.55, name, color=cam.color, fontsize=9, fontweight="bold",
                    ha="center", va="bottom")
            ax.text(cx, top + 0.5, f"{cam.resolution[0]}×{cam.resolution[1]}\n{cam.hfov_deg:g}°",
                    color=MUTED, fontsize=7, ha="center", va="top")
            f = optics.focal_px(cam)
            for di, d in enumerate(dists):
                y0 = (len(dists) - 1 - di) * row_h
                px = f * th / d
                n = max(1, int(round(px)))
                small = Image.fromarray((mask * 255).astype(np.uint8)).resize(
                    (max(1, int(round(n * aspect))), n), Image.BOX)
                big = np.asarray(small.resize((int(240 * aspect), 240), Image.NEAREST), dtype=float) / 255
                rgba = np.zeros(big.shape + (4,))
                rgba[..., :3] = to_rgba(cam.color)[:3]
                rgba[..., 3] = big
                ax.imshow(rgba, extent=(cx - hw, cx + hw, y0 + 0.28, y0 + 0.28 + fig_h),
                          interpolation="nearest", zorder=3)
                ok = round(px) >= self.setup.tiers[0].min_pixels
                ax.text(cx, y0 + 0.24, f"{px:.0f} px", color=FG if ok else COVERAGE_COLORS[0],
                        fontsize=8, ha="center", va="top")
        for di, d in enumerate(dists):
            y0 = (len(dists) - 1 - di) * row_h
            ax.text(0.05, y0 + 0.55, f"{d:g} m", color=MUTED, fontsize=9, fontweight="bold",
                    ha="left", va="center")
        ax.set_aspect("auto")
        ax.set_xlim(0, xspan)
        ax.set_ylim(-0.05, yspan)
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
