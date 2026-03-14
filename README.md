# Camera Combo Vision App

A C++ OpenCV application for a **camera + calibration + model-combination test bench**.

## Modes

- `probe` – scan camera / resolution / fps combinations and generate a health report
- `calibrate` – detect an AprilTag, compute + lock homography, define ROIs, save config
- `deploy` – run the saved homography, monitor red ROI, trigger on image ROI capture

## Project structure

```text
.
├── CMakeLists.txt
├── README.md
├── config
│   └── system_config.sample.json
├── inc
│   ├── camera.h
│   ├── calibration.h
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
cmake --build . -j
```

### Dependencies

- OpenCV **with contrib/aruco AprilTag dictionaries**
- `nlohmann_json`

On Debian / Ubuntu / Raspberry Pi OS this usually means:

```bash
sudo apt install libopencv-dev libopencv-contrib-dev nlohmann-json3-dev
```

## Run

### Probe

```bash
./camera_combo_vision_app probe --config ../config/system_config.sample.json
```

### Calibrate

```bash
./camera_combo_vision_app calibrate --config ../config/system_config.sample.json
```

### Deploy

```bash
./camera_combo_vision_app deploy --config ../config/system_config.sample.json
```

## Calibration flow

1. Open raw camera stream
2. App searches for an AprilTag according to config:
   - `family = auto` or a specific family
   - `allowed_id = -1` or a specific tag id
3. When detected, a warped top-view preview is shown
4. Press:
   - `L` to lock / unlock the current homography
   - `R` to select the **red ROI** on the warped image
   - `I` to select the **image ROI** on the warped image
   - `S` to save config immediately
   - `ESC` to exit
5. ROIs are stored as normalized ratios, not fixed pixels

## Deploy flow

Deploy does **not** redetect the tag. It only:

1. loads the saved homography
2. warps each frame
3. computes mean BGR inside the red ROI
4. triggers when:

```text
R > red_threshold
R > G + red_margin
R > B + red_margin
```

5. applies cooldown so one event does not spam every frame
6. optionally saves raw / warped / ROI images

## Speed / reliability choices baked in

- shared `open_camera()` path for probe/calibrate/deploy
- backend is configurable (`V4L2`, `ANY`, `GSTREAMER`, ...)
- optional MJPG request to reduce USB bandwidth
- `CAP_PROP_BUFFERSIZE = 1` when backend supports it
- probe warms up before measuring fps
- deploy reuses frame buffers and does no tag detection
- ROI crops are views unless a clone is actually needed for saving
- homography is saved once, then reused
- ROI ratios are clamped when reconstructed
- capture / report folders are auto-created

## Notes

- AprilTag support depends on the OpenCV build. If `DICT_APRILTAG_36h11` is missing, the OpenCV package was likely built without the relevant contrib modules.
- For Raspberry Pi camera stacks, direct `VideoCapture` behavior depends on the installed backend and pipeline. USB cameras typically work best with `CAP_V4L2`. CSI cameras may need a GStreamer path later.
