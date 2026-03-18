
# vision_app clean rewrite

This drop-in rewrite keeps the app focused on **three modes**:

- `probe`
- `calibrate`
- `deploy`

It removes the old `bench/live/tag_fill_ratio/model_path` contract and replaces it with:

- explicit `warp_width`
- explicit `warp_height`
- explicit `target_tag_px`
- backend-specific model paths for ONNX and NCNN

## Build on Raspberry Pi

```bash
export CMAKE_PREFIX_PATH=/home/pi/ncnn/build/install:$CMAKE_PREFIX_PATH

cd ~/Desktop/vision_app
rm -rf build
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

## Modes

### 1) Probe

Purpose:
- enumerate camera modes
- run a real no-UI FPS bench with the chosen mode
- optionally write a small report

Example:

```bash
./vision_app \
  --mode probe \
  --device /dev/video0 \
  --width 640 --height 480 --fourcc MJPG --fps 120 \
  --buffer-size 1 --latest-only 1 --drain-grabs 1 \
  --headless 1 --duration 5 \
  --save-report ../report/probe_report.md
```

### 2) Calibrate

Purpose:
- open the camera with the chosen mode
- detect AprilTag family `auto|16|25|36`
- map the 4 tag corners into a centered `target_tag_px x target_tag_px` square
- build and save the homography package
- edit and save `red_roi` and `image_roi`

Key idea:
- the warped image does **not** need to be tied to the source resolution
- the important geometric knob is now `target_tag_px`
- runtime scaling can be minimized by making `image_roi` match the trained model input size

Example:

```bash
./vision_app \
  --mode calibrate \
  --device /dev/video0 \
  --width 640 --height 480 --fourcc MJPG --fps 120 \
  --buffer-size 1 --latest-only 1 --drain-grabs 1 \
  --tag-family auto --target-id 0 --require-target-id 1 \
  --warp-width 384 --warp-height 384 \
  --target-tag-px 128 \
  --camera-preview-max 700 --warp-preview-max 700 \
  --save-warp ../report/warp_package.yml.gz \
  --save-rois ../report/rois.yml
```

Calibration controls:

- `SPACE / ENTER` lock current tag
- `u` unlock
- `1 / 2` select red_roi / image_roi
- `w a s d` move ROI
- `i / k` height - / +
- `j / l` width - / +
- `[` `]` move step down / up
- `,` `.` size step down / up
- `p` save warp + rois
- `y` save warp only
- `o` save rois only
- `r` reset rois
- `q / ESC` quit

### 3) Deploy

Purpose:
- load saved warp + saved rois
- compute red ratio in `red_roi`
- crop `image_roi`
- classify with ONNX or NCNN
- show result, confidence, and timing

Example (NCNN):

```bash
./vision_app \
  --mode deploy \
  --device /dev/video0 \
  --width 640 --height 480 --fourcc MJPG --fps 120 \
  --buffer-size 1 --latest-only 1 --drain-grabs 1 \
  --load-warp ../report/warp_package.yml.gz \
  --load-rois ../report/rois.yml \
  --model-enable 1 \
  --model-backend ncnn \
  --model-ncnn-param-path ../models/model.ncnn.param \
  --model-ncnn-bin-path ../models/model.ncnn.bin \
  --model-labels-path ../models/labels.txt \
  --model-input-width 128 --model-input-height 128 \
  --model-preprocess crop \
  --model-threads 4 \
  --model-stride 3 \
  --save-report ../report/deploy_report.md
```

Example (ONNX):

```bash
./vision_app \
  --mode deploy \
  --device /dev/video0 \
  --width 640 --height 480 --fourcc MJPG --fps 120 \
  --load-warp ../report/warp_package.yml.gz \
  --load-rois ../report/rois.yml \
  --model-enable 1 \
  --model-backend onnx \
  --model-onnx-path ../models/best.onnx \
  --model-labels-path ../models/labels.txt \
  --model-input-width 128 --model-input-height 128 \
  --model-preprocess crop
```

## Notes

- Keep UI args and debug args favored while tuning geometry and runtime.
- The clean design target is:
  - `ROI size == model input size`
- For real fine-tuning later, collect **real warped ROI images** from the deploy pipeline.
