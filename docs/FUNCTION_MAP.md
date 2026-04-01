# Function map

## main.cpp
- config loading
- CLI overrides
- effective config print
- mode dispatch

## camera.hpp
- probe_camera
- open_capture
- grab_latest_frame
- bench_capture

## calibrate.hpp
- detect_apriltag_best
- TagLocker
- build_warp_package_from_detection(..., center_x_ratio, center_y_ratio, ...)
- apply_warp
- save/load warp and fixed ROI yaml

## roi_helper.cpp
- extract_runtime_rois_fixed
- extract_runtime_rois_dynamic_stacked

## deploy.hpp
- AppOptions
- DynamicRedStackedConfig
- run_probe
- run_calibrate
- run_deploy
- dynamic overlay drawing
- console line builders

## text_console.hpp
- wrapped status window rendering
