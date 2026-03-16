# vision_app (5-module build)

This build stays limited to these code modules:

- `camera`
- `calibrate`
- `stats`
- `deploy`
- `main`

## What changed in this version

This version is prepared for the flow:

```text
probe camera
-> bench one camera mode
-> live AprilTag detect
-> manual/auto lock
-> compute final full-view warped preview transform
-> edit red_roi and image_roi with keyboard only
-> save H + remap cache + rois
-> deploy mode runs:
   red_roi gate -> image_roi extraction -> infer stub / later model hook
```

### Important behavior

- Live/deploy now **block oversized interactive camera modes by default**.
- The saved homography is the **final full-view warped preview transform**.
- ROIs are stored as **ratios of that final warped preview canvas**.
- Deploy uses the same saved canvas, so ROI positions stay aligned.
- Precomputed remap cache can be saved and loaded for faster runtime warp.

## Build

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

## Recommended workflow

### 1. Probe camera modes

```bash
./vision_app --mode probe
```

### 2. Benchmark one safe camera mode

```bash
./vision_app --mode bench \
  --width 1280 --height 720 --fps 30 \
  --fourcc MJPG \
  --latest-only 1 --drain-grabs 2 \
  --headless 1 --duration 10
```

### 3. Live calibration

```bash
./vision_app --mode live \
  --width 1280 --height 720 --fps 30 \
  --fourcc MJPG \
  --tag-family auto \
  --target-id 0 \
  --manual-lock-only 1 \
  --warp-width 720 --warp-height 720 \
  --warp-view-max-side 900
```

Then:

- show the tag
- press `space` to lock
- press `1` to edit `red_roi`
- press `2` to edit `image_roi`
- use `wasd` to move
- use `ijkl` to resize
- `[` `]` change move step
- `,` `.` change size step
- press `p` to save all

### 4. Deploy using saved calibration

```bash
./vision_app --mode deploy \
  --load-h ../report/warp_h.json \
  --load-rois ../report/rois.json \
  --load-remap ../report/warp_remap.yml.gz \
  --use-remap-cache 1 \
  --red-mean-threshold 120 \
  --red-dominance-threshold 20
```

## Deploy gate logic

Deploy does:

```text
warp frame
-> crop red_roi
-> if mean(red) and red dominance exceed thresholds:
      crop image_roi
      pass image_roi to infer stub
      optionally save image_roi snapshot
```

The current infer step is a lightweight placeholder that reports:

- mean gray
- stddev gray
- edge ratio

That gives you an end-to-end place to later replace with your real model call.

## Interactive size safety

Live/deploy reject large requested camera sizes by default.

Default guard:

```text
interactive_max_side = 1000
```

Override only if you deliberately want to risk instability:

```bash
--unsafe-big-frame 1
```

## Saved files

- `../report/warp_h.json`
- `../report/rois.json`
- `../report/warp_remap.yml.gz`
- `../report/latest_report.md`
- `../report/test_results.csv`
- `../report/trigger_image_roi_XXXXXX.png`
