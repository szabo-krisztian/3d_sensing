"""CLI: python -m camviz {report,preview,render} [--config ...]"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from . import optics
from .config import load_setup

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "skoda_superb_setup.yaml"


def cmd_report(setup, _args):
    tiers = setup.tiers
    print(f"{setup.vehicle.name} - target height {setup.target_height_m} m")
    head = (f"{'camera':16} {'resolution':>11} {'HFOV':>6} {'VFOV':>6} {'f [px]':>7} {'EFL':>7} "
            f"{'lens':>14} {'px/deg':>7} " + " ".join(f"{t.name[:10]:>10}" for t in tiers))
    print(head)
    print("-" * len(head))
    for c in setup.cameras:
        rs = optics.camera_ranges(setup, c)
        efl = f"{c.efl_mm:.2f}mm" if c.efl_mm else (
            f"({c.f_px * c.pixel_size_um / 1000:.2f})" if c.pixel_size_um else "-")
        kind = ("pinhole" if c.lens_a > 0.999 else "equidistant" if abs(c.lens_a) < 1e-6
                else f"barrel a={c.lens_a:+.2f}")
        print(f"{c.name:16} {c.resolution[0]:>5}x{c.resolution[1]:<5} {c.hfov_deg:6.1f} {c.vfov_deg:6.1f} "
              f"{c.f_px:7.0f} {efl:>7} {kind:>14} {optics.px_per_deg(c):7.1f} "
              + " ".join(f"{r:9.0f}m" for r in rs))
    warned = [(c.name, w) for c in setup.cameras for w in c.warnings]
    if warned:
        print("\nspec warnings:")
        for n, w in dict.fromkeys(warned):
            print(f"  {n}: {w}")
    xs, ys, masks = optics.near_field_coverage(setup, 14)
    print("\nnear-field blind gap (closest visible ground point to the body):")
    for c, mk in zip(setup.cameras, masks):
        g = optics.blind_gap(setup.vehicle, xs, ys, mk)
        print(f"  {c.name:18} {'-' if math.isinf(g) else f'{g:.2f} m'}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="camviz", description="Camera setup FOV / range video renderer")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--zoom-out", type=float,
                   help="map zoom-out factor (1 = config default, 2 = see twice as far)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("report", help="print camera ranges / coverage stats")
    pv = sub.add_parser("preview", help="render a single frame to PNG")
    pv.add_argument("--time", type=float, default=30.0, help="simulation time [s]")
    pv.add_argument("--video-time", type=float, help="video time [s] (controls the zoom-out)")
    pv.add_argument("--out", type=Path, default=Path("out/preview.png"))
    rv = sub.add_parser("render", help="render the full MP4")
    rv.add_argument("--out", type=Path)
    rv.add_argument("--max-seconds", type=float, help="only render the first N seconds of video")
    args = p.parse_args(argv)

    setup = load_setup(args.config)
    if args.cmd == "report":
        return cmd_report(setup, args)

    from .render import Renderer
    if args.zoom_out:
        setup.render.setdefault("map", {})["zoom_out"] = args.zoom_out
    r = Renderer(setup)
    if args.cmd == "preview":
        r.preview(args.out, args.time, args.video_time)
    else:
        r.render_video(args.out or setup.path(setup.render.get("output", "out/video.mp4")),
                       args.max_seconds)


if __name__ == "__main__":
    main()
