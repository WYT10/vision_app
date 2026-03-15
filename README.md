# Camera Combo App

A lightweight C++ OpenCV app for:
- USB camera probe
- tag-based homography calibration
- ROI setup
- deploy-time red trigger and model stub

## Why not `.c`?
This project uses OpenCV's C++ API (`cv::Mat`, `cv::aruco::ArucoDetector`, RAII camera handling, small OOP sessions).
That means the source files should be `.cpp` and headers `.h`.
Using `.c` would fight the library API and make the code less stable, not more lightweight.

## Structure

- `src/`
- `inc/`
- `config/`
- `reports/`
- `captures/`
- `CMakeLists.txt`

## Build

```bash
mkdir build
cd build
cmake ..
cmake --build . -j
```

## Run

```bash
./camera_combo_app new-profile --config ../config/system_config.json
./camera_combo_app probe --config ../config/system_config.json
./camera_combo_app calibrate --config ../config/system_config.json
./camera_combo_app deploy --config ../config/system_config.json
```

## Calibration controls

- `Enter`: update candidate warp preview once
- `L`: lock the current valid detection and warp
- `1`: start red ROI selection on frozen warped image
- `2`: start target ROI selection on frozen warped image
- `S`: save profile
- `Esc`: quit

## Notes

- Calibration and deploy use the same saved camera mode.
- Warp preview uses a fixed 255x255 display image, but the saved warp size is the real computed size.
- During red ROI editing, the live average red-channel value inside the ROI is shown so you can choose a threshold more confidently.
- Inference is currently a stub seam inside deploy; wire in ONNX or NCNN later.
