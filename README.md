# Vision App

A compact C++ OpenCV app for:

- **probe**: test camera modes and write a report
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
sudo apt install libopencv-dev libopencv-contrib-dev nlohmann-json3-dev
```

## Run from build/

```bash
./vision_app probe --config ../config/system_config.sample.json
./vision_app calibrate --config ../config/system_config.sample.json
./vision_app deploy --config ../config/system_config.sample.json
```

## Camera probe workflow

`probe` tries every candidate mode from config, reads back what the driver actually applied, measures actual capture fps, and writes JSON + CSV reports into `runtime.report_dir`.

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
- warmup before fps measurement
- deploy does no tag detection
- warped output buffer is reused
- ROI crops stay as views unless saving
- no unnecessary heap objects in hot deploy path

## Default active mode

The sample config uses:

```text
320x240 @ 60 fps MJPG
```

This is chosen to behave closer to your target high-speed low-resolution USB camera while still being less fragile than `160x120` during calibration.
