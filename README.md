# vision_app stable v5

Five code modules only:
- `camera.hpp`
- `calibrate.hpp`
- `stats.hpp`
- `deploy.hpp`
- `main.c`

## Purpose
- probe USB camera modes
- benchmark one camera mode
- live AprilTag calibration
- build a stable full-image auto-fit warp
- soft-limit the warp canvas so Pi does not crash on large previews
- save precomputed remap maps + valid mask + ROI config
- deploy later using the saved precomputation

## Key stabilization changes
- **single preview window** in live/deploy
- **keyboard-only ROI editing**
- **auto-fit full transformed source image** into the warp canvas
- **soft max on warp size**; large auto-fit canvases are scaled down
- **preview downscale** independent from processing warp size
- **valid mask** stored and displayed; empty/invalid regions remain visible but are ignored by later processing
- **remap cache** saved for fast later feeds

## Modes
### probe
List camera formats/resolutions/FPS from `v4l2-ctl`.

```bash
./vision_app --mode probe
```

### bench
Open one camera config and measure actual FPS.

```bash
./vision_app --mode bench --device /dev/video0 --width 1280 --height 720 --fps 30 --fourcc MJPG --duration 10
```

### live
Calibration flow:
1. open camera with your chosen config
2. detect AprilTag family `auto|16|25|36`
3. while unlocked, if a candidate exists, show temporary full-image auto-fit warp preview
4. press `space` to lock or allow auto-lock if enabled
5. edit `red_roi` and `image_roi` with keyboard
6. save homography/remap/mask/rois

Recommended start:

```bash
./vision_app --mode live \
  --device /dev/video0 \
  --width 1280 --height 720 --fps 30 \
  --fourcc MJPG \
  --tag-family auto \
  --target-id 0 \
  --require-target-id 1 \
  --manual-lock-only 1 \
  --warp-soft-max 900 \
  --preview-soft-max 600
```

### deploy
Load saved remap + ROI config and run directly.

```bash
./vision_app --mode deploy \
  --device /dev/video0 \
  --width 1280 --height 720 --fps 30 \
  --fourcc MJPG \
  --load-warp ../report/warp_package.yml.gz \
  --load-rois ../report/rois.yml
```

## Live controls
- `space` / `enter`: lock current candidate
- `u`: unlock and reacquire
- `1`: select `red_roi`
- `2`: select `image_roi`
- `w a s d`: move selected ROI
- `j l`: shrink/grow width
- `i k`: shrink/grow height
- `[` `]`: move step down/up
- `,` `.`: size step down/up
- `r`: reset selected ROI
- `p`: save all
- `o`: save ROIs only
- `y`: save warp package only
- `h`: toggle help overlay
- `q` / `ESC`: quit

## Saved files
Default under `../report/` when run from `build/`:
- `warp_package.yml.gz` : homography + remap maps + valid mask + sizes
- `rois.yml` : ratio ROIs
- `test_results.csv`
- `latest_report.md`

## Notes
- AprilTag detection uses OpenCV ArUco AprilTag dictionaries if available.
- If your OpenCV build does not include `aruco`, probe/bench still work, but live calibration and deploy warp loading from tag detection will be unavailable.
