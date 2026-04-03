
# vision_app

A cleaned first-pass C++ vision app for:
- `probe`: camera mode discovery via `v4l2-ctl`
- `live`: raw preview or headless FPS test
- `calibrate`: AprilTag lock + warp preview + trigger tuning
- `deploy`: load saved warp/profile and run trigger + classifier

This package keeps the current working atoms, but reorganizes the project around a cleaner profile/artifact split and a separate status window so numeric/debug text does not fight the preview bounds. It also supports both local V4L2 cameras and RTSP/HTTP streams through a split camera implementation.

## Project layout

```text
vision_app/
├── CMakeLists.txt
├── README.md
├── cmake/
│   └── FindONNXRuntime.cmake
├── config/
│   └── profile.example.conf
├── report/
└── src/
    ├── app_config.hpp
    ├── app_config.cpp
    ├── app_types.hpp
    ├── camera.hpp
    ├── camera.cpp
    ├── calibrate.hpp
    ├── classifier_common.hpp
    ├── deploy_runtime.hpp
    ├── main.cpp
    ├── model.hpp
    ├── model.cpp
    ├── ncnn_classifier.hpp
    ├── onnx_classifier.hpp
    ├── roi_helper.cpp
    ├── stats.hpp
    ├── status_ui.hpp
    ├── status_ui.cpp
    ├── trigger.hpp
    └── trigger.cpp
```

## Build

Use this exactly:

```bash
export CMAKE_PREFIX_PATH=/home/pi/ncnn/build/install:$CMAKE_PREFIX_PATH

cd ~/Desktop/vision_app
rm -rf build
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

## Runtime model

There are two saved layers:

1. **Profile config**: `config/profile.conf`
   - camera assumptions
   - tag lock policy
   - trigger mode and parameters
   - red thresholds
   - model settings
   - UI/text routing behavior

2. **Derived warp artifact**: `report/warp_package.yml.gz`
   - homography
   - remap maps
   - valid mask
   - source size / warp size
   - locked tag family/id

This keeps the editable intent separate from the expensive precomputed warp.

## Trigger modes

### `fixed_rect`
- tunes `red_roi` and `image_roi`
- reports live `red_ratio`
- triggers when `red_ratio >= red_ratio_threshold`

### `dynamic_red_stacked`
- tunes `upper_band`, `lower_band`, and derived image ROI size/offset
- extracts a centerline from the red bands
- first-pass derived image ROI is anchored **above the upper band**
- this is a clean starting point, not a final tuned policy for every field layout

## Text routing / overlays

`text_sink` controls where debug information goes:
- `overlay`
- `status_window`
- `terminal`
- `split`

Recommended default is `split`.

This keeps image windows for geometry and uses the status window for numeric state.

## Example commands

```bash
./vision_app --mode probe --device /dev/video0
./vision_app --mode live --device /dev/video0 --width 160 --height 120 --fps 120
./vision_app --mode calibrate --config config/profile.conf
./vision_app --mode deploy --config config/profile.conf
```

## Calibration controls

- `enter` / `space`: lock current AprilTag warp
- `u`: unlock
- `p`: save profile + rois + warp + report
- `y`: save warp only
- `o`: save profile only
- `q` / `esc`: quit
- `[` `]`: move step down / up
- `,` `.`: size step down / up

### fixed_rect
- `1`: edit `red_roi`
- `2`: edit `image_roi`
- `w a s d`: move
- `i k`: height - / +
- `j l`: width - / +

### dynamic_red_stacked
- `1`: edit `upper_band`
- `2`: edit `lower_band`
- `3`: edit derived image ROI config
- `w s`: move band or ROI bottom offset
- `i k`: height - / +
- `j l`: width - / +

## Known limits of this first-pass clean version

- The dynamic mode is intentionally conservative. It is structurally clean and inspectable, but the derived ROI anchoring rule is still a first-pass assumption.
- The fixed-rect path is the more mature path.
- The project was reorganized for clarity and future iteration; more tuning keys and config fields can be added without re-entangling the core loops.
