# vision_app (5-module build)

This layout keeps the code limited to these modules:

- `camera`
- `calibrate`
- `stats`
- `deploy`
- `main`

Implemented as:

```text
vision_app_min5/
├── CMakeLists.txt
├── README.md
├── vision_app.conf
├── report/
└── src/
    ├── main.c
    ├── camera.hpp
    ├── calibrate.hpp
    ├── stats.hpp
    └── deploy.hpp
```

The build compiles only `src/main.c`. The other four modules are included as headers, so the file count stays low.

## What it can do

### `--mode probe`
- query `v4l2-ctl`
- normalize USB camera modes
- print a compact console table
- write `probe_table.csv`

### `--mode bench`
- open the selected camera mode
- favor the newest frame (`buffer_size=1`, optional drain grabs)
- measure actual capture FPS and frame timing
- write `test_results.csv`

### `--mode live`
- live camera preview
- detect AprilTag
- show tag family / id / corners
- lock stable tag across N frames
- compute homography from **tag quadrilateral itself**
- warp the full frame
- overlay and edit two normalized ROIs:
  - `red_roi`
  - `image_roi`
- save `warp_h.json` and `rois.json`

### `--mode deploy`
- load saved homography + ROIs
- skip AprilTag acquisition
- warp immediately
- crop the two ROIs every frame

## Build

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

## Important dependency note

The AprilTag path in this package uses **OpenCV ArUco AprilTag dictionaries** when `<opencv2/aruco.hpp>` is available.

If your Pi build does **not** include OpenCV aruco/contrib, the project will still compile, but `live` mode will report that AprilTag support is unavailable until you install an AprilTag-capable backend.

On Raspberry Pi OS / Debian-style systems, that often means installing the contrib package in addition to `libopencv-dev`.

## Good first commands

Probe camera:

```bash
./vision_app --mode probe
```

Benchmark candidate camera mode:

```bash
./vision_app --mode bench \
  --device /dev/video0 \
  --width 1280 --height 720 --fps 30 \
  --fourcc MJPG \
  --latest-only 1 --drain-grabs 2 \
  --headless 1 --duration 10
```

Live AprilTag -> lock -> warp -> ROI edit:

```bash
./vision_app --mode live \
  --device /dev/video0 \
  --width 1280 --height 720 --fps 30 \
  --fourcc MJPG \
  --tag-family tag36h11 \
  --target-id 0 \
  --warp-width 1280 --warp-height 720
```

Deploy from saved calibration:

```bash
./vision_app --mode deploy \
  --device /dev/video0 \
  --width 1280 --height 720 --fps 30 \
  --fourcc MJPG \
  --load-h ../report/warp_h.json \
  --load-rois ../report/rois.json
```

## Live mode keys

- `q` / `ESC`: quit
- `p`: save homography + ROIs
- `u`: unlock and reacquire tag
- `1`: select `red_roi`
- `2`: select `image_roi`
- `w a s d`: move selected ROI
- `i j k l`: resize selected ROI
- `r`: reset ROIs to defaults
- `c`: save warped snapshot (when enabled)

## Why this structure is useful

The main system split is:

```text
camera capture
-> calibration / AprilTag lock
-> warp full frame
-> normalized ROIs
-> downstream processing
```

That means you can first solve:

- camera mode selection
- frame freshness
- AprilTag lock quality
- warp stability

Then later add your downstream logic on top of the warped `red_roi` and `image_roi` without rewriting the geometry layer.
