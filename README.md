# vision_app working drop-in

This drop-in keeps the existing camera + AprilTag + ROI workflow and replaces the old model stub with the proven ONNX Runtime / NCNN classifier runtime.

## What changed

- keeps the runtime app focused on three modes:
  - `probe`
  - `calibrate`
  - `deploy`
- uses explicit calibration knobs:
  - `--warp-width`
  - `--warp-height`
  - `--target-tag-px`
- integrates the working classifier backends from the standalone benchmark helpers:
  - ONNX Runtime
  - NCNN
- removes the old OpenCV DNN ONNX stub and the fake NCNN hook
- adds deploy warnings when the live camera mode does not match the saved calibration source size

The standalone classifier helpers you shared already support repeated latency and report generation, and they show the same ONNX/NCNN runtime pattern this app now uses. fileciteturn26file1 fileciteturn26file2 fileciteturn26file3

## Files in this drop-in

Replace your project with these files/folders:

- `src/main.c`
- `src/camera.hpp`
- `src/calibrate.hpp`
- `src/deploy.hpp`
- `src/roi_helper.cpp`
- `src/model.hpp`
- `src/model.cpp`
- `src/classifier_common.hpp`
- `src/onnx_classifier.hpp`
- `src/ncnn_classifier.hpp`
- `src/stats.hpp`
- `cmake/FindONNXRuntime.cmake`
- `CMakeLists.txt`
- `README.md`

## Pull latest repo first

```bash
git pull https://github.com/WYT10/vision_app.git main
```

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

Probe supported camera formats and run a short real FPS test.

```bash
./vision_app --mode probe \
  --device /dev/video0 \
  --width 640 --height 480 --fourcc MJPG --fps 120 \
  --duration 5
```

### 2) Calibrate

Detect the AprilTag, map it to a centered `target_tag_px x target_tag_px` square inside `warp_width x warp_height`, tune ROIs, then save warp and ROI config.

```bash
./vision_app --mode calibrate \
  --device /dev/video0 \
  --width 640 --height 480 --fourcc MJPG --fps 120 \
  --tag-family auto --target-id 0 \
  --warp-width 384 --warp-height 384 \
  --target-tag-px 128 \
  --save-warp ../report/warp_package.yml.gz \
  --save-rois ../report/rois.yml \
  --save-report ../report/latest_report.md
```

Controls:
- `SPACE` / `ENTER`: lock current tag
- `u`: unlock
- `1` / `2`: select red ROI / image ROI
- `w a s d`: move ROI
- `i k j l`: resize ROI
- `[` `]` `,` `.`: step size tuning
- `o`: save ROIs
- `y`: save warp
- `p`: save all
- `q` / `ESC`: quit

### 3) Deploy

Load the saved warp + ROIs, compute the red ROI ratio, crop the image ROI, and run classification.

#### ONNX

```bash
./vision_app --mode deploy \
  --device /dev/video0 \
  --width 640 --height 480 --fourcc MJPG --fps 120 \
  --load-warp ../report/warp_package.yml.gz \
  --load-rois ../report/rois.yml \
  --model-enable 1 \
  --model-backend onnx \
  --model-onnx-path ../models/best.onnx \
  --model-labels-path ../models/labels.txt \
  --model-input-width 128 --model-input-height 128 \
  --model-preprocess crop \
  --model-threads 4 \
  --model-stride 1
```

#### NCNN

```bash
./vision_app --mode deploy \
  --device /dev/video0 \
  --width 640 --height 480 --fourcc MJPG --fps 120 \
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
  --model-stride 1
```

## Important notes

### Match deploy camera mode to calibration camera mode

The saved warp package stores the original calibration source size. If deploy opens the camera with a different width/height, the app now warns you because the warp may no longer match.

### ROI size should match model input size

The cleanest setup is:

- `target_tag_px` chosen during calibration
- ROI tuned in warped space
- `image_roi` content naturally lands at the same scale the classifier expects
- model input size matches the training/deploy size

### Current benchmark observation

From your Pi tests on the standalone classifier, NCNN is already faster than ONNX at `128x128` for the same image:
- ONNX: ~34 ms/img
- NCNN: ~13 ms/img

So NCNN should be your default deploy backend on Pi unless you specifically need ONNX parity/debug.

## Model reference

The working classifier pieces in this drop-in are based on the standalone runtime you shared:
- shared preprocessing and label handling in `classifier_common.hpp` fileciteturn26file0
- ONNX Runtime classifier with quiet load behavior in `onnx_classifier.hpp` fileciteturn26file3
- NCNN classifier with robust input/output blob detection in `ncnn_classifier.hpp` fileciteturn26file2
- matching build logic in the standalone CMake file fileciteturn26file5

## Dataset note

The standalone benchmark helpers work directly on the `dataset_cls/train`, `dataset_cls/val`, and `dataset_cls/test` folder structure without a YAML file, using the parent folder name as ground truth. fileciteturn26file4
