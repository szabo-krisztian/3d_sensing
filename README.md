# camviz — camera setup FOV / range video

Renders a presentation video (MP4, 1080p) of a vehicle camera setup driving a real route on
OpenStreetMap, showing:

- **Map chase view** (heading-up; it zooms out from ~90 m to ~330 m at the start). Every
  camera's FOV is drawn out to its detection range and fades as pixel density drops.
- **Near-field coverage**: how many cameras see each ground cell around the car,
  including occlusion by a coarse car body model, so blind spots appear in red.
- **Range by resolution**: top view; solid edge = 20 px on a pedestrian, dotted = 50 / 120 px.
- **Pixel resolution**: a pedestrian rendered at its true pixel height for each camera
  at 10–200 m, so you can see the pixelation.

Ranges are derived from resolution: the distance at which a target of `target_height_m`
spans `min_pixels` vertical pixels (`range = f_px * H / N`, at the image centre).

## Usage

```bash
pip install -r requirements.txt          # numpy, matplotlib, pillow, pyyaml, requests (+ ffmpeg)
python3 -m camviz report                 # print ranges / blind gaps
python3 -m camviz preview --time 60      # single frame -> out/preview.png
python3 -m camviz render                 # full video -> out/camera_setup_drive.mp4
python3 -m camviz render --max-seconds 5 --out out/test.mp4
python3 -m camviz --zoom-out 1.5 render  # map shows 1.5x further (also: render.map.zoom_out)
python3 -m camviz --config my_setup.yaml render
```

## Configuring

Edit `config/skoda_superb_setup.yaml`:

- `camera_models`: the spec catalog (model, resolution, HFOV/VFOV, `pinhole`/`fisheye`).
- `cameras`: placements. Each references a model and gives its mount position and
  yaw/pitch/roll in the ISO 8855 vehicle frame (origin at the centre of the rear axle,
  x forward, y left, z up). Any model field can be overridden per camera.

Long-range and mid-range specs come from the slides and spreadsheet. The fisheye
(SENSING SG3S-ISX031C-GMSL2F) and rear (Leopard LI-IMX728-9295-120H) models are
assumptions, because the slides do not specify them. Mount positions are estimated
from the slide-1 overview.

The route is set by `trajectory.waypoints` (lat/lon). These are snapped to roads with the
public OSRM router and cached in `data/route_cache.json`. Map tiles are cached in
`data/tile_cache/`. After the first run, everything works offline.
