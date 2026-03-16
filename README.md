# vision_app live-safe v6

This version keeps the project to 5 code modules:

- `camera.hpp`
- `calibrate.hpp`
- `stats.hpp`
- `deploy.hpp`
- `main.c`

## What changed in this version

The live preview path is made safer for Raspberry Pi:

- one main window only
- **searching mode** shows raw frame with tag overlay + a **small inset live warp preview**
- the inset preview is rebuilt only every `temp_preview_stride` frames
- the inset preview uses a small capped square (`temp_preview_square`)
- after lock, the app shows the **locked warped preview** in the same main window
- keyboard-only ROI editing
- auto-fit warp of the transformed full image
- soft limits clamp camera size, warp size, and preview size
- invalid warped pixels are tracked by a mask and shown on a white canvas

## Main functions now

### probe
List camera formats, resolutions, and advertised FPS.

```bash
./vision_app --mode probe
```

### bench
Measure real runtime FPS for one camera configuration.

Good starting command:

```bash
./vision_app --mode bench \
  --device /dev/video0 \
  --width 640 --height 480 \
  --fourcc MJPG \
  --fps 180 \
  --buffer-size 1 \
  --latest-only 1 \
  --drain-grabs 1 \
  --headless 1 \
  --duration 10
```

### live
Live calibration with safer warp preview.

Good starting command:

```bash
./vision_app --mode live \
  --device /dev/video0 \
  --width 640 --height 480 \
  --fourcc MJPG \
  --fps 180 \
  --buffer-size 1 \
  --latest-only 1 \
  --drain-grabs 1 \
  --tag-family auto \
  --target-id 0 \
  --require-target-id 1 \
  --manual-lock-only 1 \
  --warp-soft-max 700 \
  --preview-soft-max 500
```

### deploy
Load saved warp + ROIs and run directly.

```bash
./vision_app --mode deploy \
  --device /dev/video0 \
  --width 640 --height 480 \
  --fourcc MJPG \
  --fps 180 \
  --load-warp ../report/warp_package.yml.gz \
  --load-rois ../report/rois.yml
```

## Live controls

### Searching / live preview
- `space` or `enter`: lock current visible tag
- `u`: unlock and go back to searching
- `h`: toggle help overlay
- `q` or `ESC`: quit

### ROI editing after lock
- `1`: select `red_roi`
- `2`: select `image_roi`
- `w a s d`: move selected ROI
- `i k`: change ROI height
- `j l`: change ROI width
- `[` `]`: smaller / larger move step
- `,` `.`: smaller / larger size step
- `r`: reset ROIs to defaults
- `p`: save warp + rois + report
- `o`: save rois only
- `y`: save warp only

## Notes

- Missing ROI file on startup is treated as normal and defaults are used.
- If requested camera size exceeds `camera_soft_max`, the code clamps it down automatically.
- If fitted warp size exceeds `warp_soft_max`, the code scales it down automatically.
- Preview is independently downscaled to `preview_soft_max`.
- Invalid warped regions are shown in white and saved in the mask inside the warp package.
