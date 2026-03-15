# Vision App

A compact C++ OpenCV app for:

- **probe**: enumerate USB camera capabilities and run quick real capture tests
- **calibrate**: detect AprilTag, lock homography, select ROIs, save config
- **deploy**: reuse the saved homography and ROI trigger path

## Design choice locked in

Calibration and deploy use the **same active camera mode** from config.
That keeps the homography, warped size, and ROI ratios consistent.

## Project structure

```text
.
├── CMakeLists.txt
├── README.md
├── config
│   └── system_config.sample.json
├── inc
│   ├── calibration.h
│   ├── camera.h
│   ├── config.h
│   ├── deploy.h
│   └── homography.h
└── src
    ├── calibration.cpp
    ├── camera.cpp
    ├── config.cpp
    ├── deploy.cpp
    ├── homography.cpp
    └── main.cpp
```

## Build

```bash
mkdir build
cd build
cmake ..
make -j4
```

## Dependencies

```bash
sudo apt install libopencv-dev libopencv-contrib-dev nlohmann-json3-dev v4l-utils
```

## Run from build/

```bash
./vision_app probe --config ../config/system_config.sample.json
./vision_app calibrate --config ../config/system_config.sample.json
./vision_app deploy --config ../config/system_config.sample.json
```

## Probe behavior

The probe now has **two layers**:

1. **Driver enumeration layer**
   - runs the same V4L2 commands you would use manually
   - stores the raw output in the report bundle
2. **Real capture test layer**
   - requests each candidate mode via OpenCV
   - measures actual capture FPS after warmup
   - records requested vs actual mode

Generated files:

- `camera_probe_*.json`
- `camera_probe_*.csv`
- `camera_probe_*_v4l2.txt`

Recommended first step on a new USB camera:

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
./vision_app probe --config ../config/system_config.sample.json
```

Then pick one usable mode from the report and copy it into `camera.requested_mode`.

## Calibration controls

- `L` lock / unlock homography
- `R` select red ROI on warped image
- `I` select image ROI on warped image
- `S` save config
- `ESC` exit

## Embedded-oriented choices already applied

- one active camera mode for calibration + deploy
- probe only measures candidate modes; deploy never changes mode
- camera open path is shared for probe / calibration / deploy
- MJPG can be requested through config to reduce USB bandwidth
- buffer size hint is set to 1 when supported
- warmup before FPS measurement
- deploy does no tag detection
- warped output buffer is reused
- ROI crops stay as views unless saving
- unsafe warp sizes are rejected before allocation
- no unnecessary heap objects in hot deploy path

## Default active mode

The sample config uses:

```text
320x240 @ 60 fps MJPG
```

This is chosen to behave closer to your target high-speed low-resolution USB camera while still being less fragile than `160x120` during calibration.
