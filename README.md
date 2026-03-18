# vision_app drop-in patch

This patch replaces the old OpenCV-DNN-only model path with the tested ONNX Runtime / NCNN classifier runtime and updates the main app build so `vision_app` can classify `image_roi` directly in `live` and `deploy`.

## Replace/add these files

Copy the contents of this patch over your project root:

- `CMakeLists.txt`
- `README.md`
- `cmake/FindONNXRuntime.cmake`
- `src/main.c`
- `src/deploy.hpp`
- `src/roi_helper.cpp`
- `src/model.cpp`
- `src/classifier_common.hpp`
- `src/onnx_classifier.hpp`
- `src/ncnn_classifier.hpp`

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

If ONNX Runtime is already installed in a standard system path, the app should find it automatically through `FindONNXRuntime.cmake`.

## What changes compared to the old app

- keeps executable name `vision_app`
- keeps current modes: `probe | bench | live | deploy`
- `live` can classify the locked `image_roi`
- `deploy` can classify `image_roi` every `model_stride` frames
- ONNX Runtime backend supported in main app
- NCNN backend supported in main app
- labels file supported in main app
- preprocess mode supported: `crop | stretch | letterbox`

## Model config in `vision_app.conf`

```ini
model_enable=1
model_backend=ncnn
model_onnx_path=../models/best.onnx
model_ncnn_param_path=../models/model.ncnn.param
model_ncnn_bin_path=../models/model.ncnn.bin
model_labels_path=../models/labels.txt
model_input_width=224
model_input_height=224
model_threads=4
model_topk=5
model_stride=5
model_preprocess=crop
model_quiet_onnx_load=1
model_mean=0,0,0
model_norm=0.0039215686,0.0039215686,0.0039215686
```

## Suggested runtime rule

Use the same ROI size and preprocessing contract you trained with on the computer.

If your classifier was trained with `224x224` crop-style preprocessing, keep:

- `model_input_width=224`
- `model_input_height=224`
- `model_preprocess=crop`

## Example: live calibration / ROI edit

```bash
./vision_app \
  --mode live \
  --device /dev/video0 \
  --width 640 --height 480 --fourcc MJPG --fps 120 \
  --latest-only 1 --drain-grabs 1 \
  --tag-family auto \
  --target-id 0 \
  --manual-lock-only 1 \
  --camera-preview-max 640 \
  --warp-preview-max 640
```

## Example: deploy with NCNN

```bash
./vision_app \
  --mode deploy \
  --device /dev/video0 \
  --width 640 --height 480 --fourcc MJPG --fps 120 \
  --load-warp ../report/warp_package.yml.gz \
  --load-rois ../report/rois.yml \
  --model-enable 1 \
  --model-backend ncnn \
  --model-ncnn-param-path ../models/model.ncnn.param \
  --model-ncnn-bin-path ../models/model.ncnn.bin \
  --model-labels-path ../models/labels.txt \
  --model-input-width 224 \
  --model-input-height 224 \
  --model-preprocess crop \
  --model-threads 4
```

## Notes

- `red_roi` and `image_roi` stay ratio rectangles in warp space.
- invalid warped pixels stay white and are masked out during ROI operations.
- the old OpenCV-DNN model stub is removed from the main app.
- for Raspberry Pi deployment, NCNN is the preferred backend.
