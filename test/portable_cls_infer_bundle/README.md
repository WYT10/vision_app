# Portable image-classification inference bundle (ONNX + NCNN)

This bundle is a small C++17 project for running your exported **Ultralytics classification** model on desktop or edge targets.

It includes:

- `src/onnx_classifier.hpp` — ONNX Runtime backend
- `src/ncnn_classifier.hpp` — NCNN backend
- `src/main.cpp` — CLI for single-image or folder inference
- `cmake/FindONNXRuntime.cmake` — helper for locating a prebuilt ONNX Runtime package
- `CMakeLists.txt` — cross-platform build entry point

## What this is for

Use it after you already have:

- `best.onnx`
- `best.param`
- `best.bin`
- `labels.txt`

Training on a GPU desktop does **not** prevent deployment to CPU-only boards. The deployment target only needs a runtime that can execute the exported model.

## Preprocessing defaults

The default CLI preprocessing is:

- `--prep crop`
- RGB conversion
- `mean = 0,0,0`
- `norm = 1/255,1/255,1/255`

That matches current Ultralytics classify inference much more closely than a raw stretch path.

If you retrained with a custom resize-only pipeline, switch to:

- `--prep stretch`

### Summary of modes

- `crop` — center-crop largest square, then resize to `imgsz`
- `stretch` — direct resize to `imgsz x imgsz`
- `letterbox` — pad to square, then resize

## Project tree

```text
portable_cls_infer_bundle/
├─ CMakeLists.txt
├─ README.md
├─ cmake/
│  └─ FindONNXRuntime.cmake
└─ src/
   ├─ classifier_common.hpp
   ├─ onnx_classifier.hpp
   ├─ ncnn_classifier.hpp
   └─ main.cpp
```

## Build dependencies

### Common

- CMake >= 3.16
- C++17 compiler
- OpenCV (`core`, `imgproc`, `imgcodecs`)

### ONNX backend

You need a prebuilt ONNX Runtime C/C++ package or your own build.
Point CMake at it with:

- `-DONNXRUNTIME_ROOT=/path/to/onnxruntime`

The folder should contain something like:

```text
onnxruntime/
├─ include/
│  └─ onnxruntime_cxx_api.h
└─ lib/
   └─ onnxruntime.lib / libonnxruntime.so
```

### NCNN backend

Build/install NCNN first, then point CMake at the directory containing `ncnnConfig.cmake`:

- `-Dncnn_DIR=/path/to/ncnn/install/lib/cmake/ncnn`

## Configure and build

### Desktop build with both backends

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_ONNX_RUNTIME=ON \
  -DENABLE_NCNN=ON \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
  -Dncnn_DIR=/path/to/ncnn/install/lib/cmake/ncnn

cmake --build build --config Release
```

### Raspberry Pi / CPU-only board (NCNN only)

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_ONNX_RUNTIME=OFF \
  -DENABLE_NCNN=ON \
  -Dncnn_DIR=/home/pi/ncnn/build/install/lib/cmake/ncnn

cmake --build build -j4
```

## Running the CLI

### ONNX single image

```bash
./portable_cls_infer \
  --backend onnx \
  --model /path/to/best.onnx \
  --labels /path/to/labels.txt \
  --input /path/to/test.jpg \
  --prep crop
```

### NCNN single image

```bash
./portable_cls_infer \
  --backend ncnn \
  --model /path/to/best.param \
  --weights /path/to/best.bin \
  --labels /path/to/labels.txt \
  --input /path/to/test.jpg \
  --prep crop \
  --threads 4
```

### Folder test on exported runtime

If your folder structure is:

```text
dataset_cls/test/
├─ A_gun/
├─ B_explosive/
└─ ...
```

then you can do a quick runtime-side accuracy check:

```bash
./portable_cls_infer \
  --backend ncnn \
  --model /path/to/best.param \
  --weights /path/to/best.bin \
  --labels /path/to/labels.txt \
  --input /path/to/dataset_cls/test \
  --eval-parent-label
```

This treats the image parent folder name as the expected class label.

## Recommended deployment path

### PC smoke test

1. Test `best.onnx` with the ONNX backend
2. Confirm labels and preprocessing match
3. Compare against your Python/Ultralytics predictions

### Pi deployment

1. Copy `best.param`, `best.bin`, and `labels.txt` to the Pi
2. Build NCNN on the Pi
3. Build this project with `-DENABLE_ONNX_RUNTIME=OFF -DENABLE_NCNN=ON`
4. Run folder or image tests
5. Integrate the NCNN header into your camera/homography pipeline

## Practical notes

- For Raspberry Pi, start with `--threads 4` on Pi 5 and tune later.
- Keep Vulkan off first. Get CPU inference stable before exploring GPU/Vulkan.
- If predictions differ from Python, the first thing to check is preprocessing:
  - `crop` vs `stretch`
  - RGB/BGR handling
  - mean/norm values
- `labels.txt` must match the class order used during training/export.

## Typical integration plan into your app

Once this CLI works, the next step is usually:

1. load model once at startup
2. capture frame
3. rectify / warp top view
4. classify warped ROI
5. draw best label + probability on the frame

That means:

- geometry is still handled by your OpenCV / AprilTag / homography code
- recognition is handled by the exported ONNX or NCNN model

## Notes about this bundle

This project is intentionally small and header-heavy so you can either:

- build it as a standalone test executable, or
- copy the backend header into your existing app and call it directly
