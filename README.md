# vision_app (5-module build)

This layout stays limited to these code modules:

- `camera`
- `calibrate`
- `stats`
- `deploy`
- `main`

Implemented as:

```text
vision_app_min5_v4_remap/
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

## Core workflow

```text
probe camera
-> bench candidate camera modes
-> choose one camera config
-> live AprilTag detection
-> show family + id
-> lock the tag
-> compute homography from tag quadrilateral itself
-> warp the full frame
-> set ratio rois in warped coordinates
-> save homography + rois
-> deploy mode loads them directly later
```

## Mode summary

### `--mode probe`
Use this first.

What it does:
- calls `v4l2-ctl`
- normalizes repeated mode listings
- prints a compact resolution / fps table
- writes `probe_table.csv`

Typical command:

```bash
./vision_app --mode probe --device /dev/video0
```

---

### `--mode bench`
Use this to decide which camera mode is actually good enough for AprilTag and warp.

What it does:
- opens one requested camera mode
- measures actual capture fps and timing
- favors latest-frame behavior when configured
- appends one row into `test_results.csv`

Most useful arguments:
- `--width --height --fps --fourcc`
- `--buffer-size`
- `--latest-only`
- `--drain-grabs`
- `--warmup`
- `--duration`
- `--headless`

Example:

```bash
./vision_app --mode bench \
  --device /dev/video0 \
  --width 1280 --height 720 --fps 30 \
  --fourcc MJPG \
  --latest-only 1 --drain-grabs 2 --buffer-size 1 \
  --headless 1 --duration 10
```

Interpretation rule:
- if bench fps is low or unstable, live AprilTag lock will also be weak
- choose the mode that gives enough image detail **and** enough real fps headroom

---

### `--mode live`
Use this for calibration and ROI setup.

What it does:
- runs live AprilTag detection
- supports family search modes: `auto`, `16`, `25`, `36`
- shows detected family / id / corners
- locks the tag automatically or manually
- computes homography from the **tag quadrilateral itself**
- warps the **full frame**
- lets you edit `red_roi` and `image_roi` as ratio `x,y,w,h`
- saves `warp_h.json` and `rois.json`

Family options:
- `--tag-family auto` searches `36 -> 25 -> 16`
- `--tag-family 16` uses `tag16h5`
- `--tag-family 25` uses `tag25h9`
- `--tag-family 36` uses `tag36h11`

Lock options:
- `--require-target-id 1` means only the specified id can lock
- `--manual-lock-only 1` means never auto-lock; press `space` or `enter`
- `--lock-frames N` controls the auto-lock stability requirement

ROI options:
- `--red-roi x,y,w,h`
- `--image-roi x,y,w,h`
- `--move-step X`
- `--size-step X`

Preview options:
- `--live-preview-raw 0|1`
- `--live-preview-warp 0|1`
- `--show-roi-crops 0|1`
- `--show-help-overlay 0|1`
- `--save-snapshots 0|1`

Example:

```bash
./vision_app --mode live \
  --device /dev/video0 \
  --width 1280 --height 720 --fps 30 \
  --fourcc MJPG \
  --tag-family auto \
  --target-id 0 --require-target-id 1 \
  --lock-frames 8 \
  --warp-width 1280 --warp-height 720 \
  --red-roi 0.05,0.10,0.20,0.20 \
  --image-roi 0.30,0.10,0.50,0.60
```

Live keys:
- `q` / `ESC` quit
- `h` toggle help overlay
- `space` / `enter` force lock current visible tag
- `u` unlock and reacquire
- `p` save all outputs
- `y` save homography only
- `o` save rois only
- `t` toggle auto-save on lock
- `1` / `2` select `red_roi` / `image_roi`
- `TAB` switch selected roi
- `w a s d` move selected roi
- `i / k` shrink / grow roi height
- `j / l` shrink / grow roi width
- `z / x` decrease / increase move step
- `n / m` decrease / increase size step
- `r` reset rois to defaults
- `c` save warped snapshot

---

### `--mode deploy`
Use this after calibration when the camera and plane stay fixed.

What it does:
- loads saved homography and rois
- skips AprilTag acquisition
- warps the full frame immediately
- shows warped output and ROI crops

Example:

```bash
./vision_app --mode deploy \
  --device /dev/video0 \
  --width 1280 --height 720 --fps 30 \
  --fourcc MJPG \
  --load-h ../report/warp_h.json \
  --load-rois ../report/rois.json
```

## Build

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

## Dependency note

The AprilTag path in this package uses **OpenCV ArUco AprilTag dictionaries** when `<opencv2/aruco.hpp>` is available.

If your Pi build does not include OpenCV aruco/contrib, probe and bench still work, but live AprilTag mode will report that AprilTag support is unavailable until you install an AprilTag-capable backend.


## Faster warp path

After calibration, the app can precompute a remap cache and save it to `../report/warp_remap.yml.gz`.
Later deploy runs can load that cache so the per-frame warp uses `cv::remap()` with precomputed maps instead of recomputing geometry each frame.

Useful flags:

```bash
--use-remap-cache 1
--fixed-point-remap 1
--save-remap ../report/warp_remap.yml.gz
--load-remap ../report/warp_remap.yml.gz
```
