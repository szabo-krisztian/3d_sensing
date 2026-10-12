# camviz

Renders a 1080p presentation video of the Skoda Superb camera setup driving a real
OpenStreetMap route, showing each camera's FOV, range and pixel resolution.

## Render

```bash
pip install -r requirements.txt     # also needs ffmpeg
python3 -m camviz --zoom-out 1.5 render
```

The output is `out/camera_setup_drive.mp4`. A render takes about 15 minutes. The first run
needs internet access to download the map tiles.

Quick checks before a full render:

```bash
python3 -m camviz preview --time 60       # one frame -> out/preview.png
python3 -m camviz report                  # ranges and spec warnings
```

## Important files

| File | What it is |
|---|---|
| `config/skoda_superb_setup.yaml` | **Edit this.** Camera models, mount positions, route, video settings |
| `camviz/optics.py`, `camviz/lens.py` | Projection, lens model, range calculation |
| `camviz/render.py` | Video layout and drawing |
| `tests/verify_geometry.py` | Numerical checks of the geometry |

Positions in the config use the vehicle frame: origin at the centre of the rear axle,
x forward, y left, z up, in metres. Range is the distance at which a 1.75 m pedestrian
still spans 20 px (detection), 50 px (recognition) or 120 px (identification).

The fisheye and rear camera models are assumptions, and the mount positions are
estimated from the slides.
