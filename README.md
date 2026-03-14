# Camera + Calibration + Model Test App (Modular C++)

This version is restructured into a simpler project layout:

```text
./src
./inc
CMakeLists.txt
README.md
```

It splits the app into smaller files so the system is easier to read, debug, and extend.

## What each file does

### Core app flow
- `src/main.cpp` — process entry point
- `src/app.cpp` — top-level mode dispatch: `probe`, `calibrate`, `deploy`
- `src/cli.cpp` / `inc/cli.hpp` — command-line parsing and config overrides

### Shared data / config
- `inc/types.hpp` — all shared structs and state containers
- `src/config.cpp` / `inc/config.hpp` — JSON load/save for system config
- `src/utils.cpp` / `inc/utils.hpp` — timestamp, directory creation, string helpers
- `inc/constants.hpp` — window names and fixed constants

### Camera / probing
- `src/camera.cpp` / `inc/camera.hpp` — open camera, backend mapping, frame read
- `src/probe.cpp` / `inc/probe.hpp` — camera index + resolution + fps scan and report export

### Tag + warp
- `src/tag_detector.cpp` / `inc/tag_detector.hpp` — AprilTag family selection and detection filtering
- `src/warp.cpp` / `inc/warp.hpp` — homography build, warp, ROI normalization helpers

### Calibration UI
- `src/roi_selector.cpp` / `inc/roi_selector.hpp` — ROI drawing on warped frame
- `src/calibration.cpp` / `inc/calibration.hpp` — live calibration loop, lock transform, save ROIs

### Deploy runtime
- `src/deploy.cpp` / `inc/deploy.hpp` — load calibration, monitor red ROI, crop image ROI on trigger

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

Dependencies:
- OpenCV with `aruco`, `videoio`, `highgui`, `imgproc`, `imgcodecs`
- `nlohmann_json`

## Run

### 1) Probe cameras
```bash
./camera_combo_app probe --config config/system_config.json
```

This writes JSON and CSV reports into `./reports`.

### 2) Calibrate
```bash
./camera_combo_app calibrate --config config/system_config.json
```

Controls:
- `L` lock current homography
- `R` draw red ROI on warped view
- `I` draw image ROI on warped view
- `S` save config
- `Q` or `ESC` quit

### 3) Deploy
```bash
./camera_combo_app deploy --config config/system_config.json
```

Deploy behavior:
- use saved homography
- check average red ratio in `red_roi`
- when threshold passes, capture `image_roi`
- optionally save raw / warped / ROI images

## Notes on structure

This is intentionally organized around **responsibility boundaries**:
- camera IO
- perception/tag detection
- geometric transform
- calibration interaction
- deployment logic

That separation makes it easier to replace pieces later, for example:
- swap AprilTag implementation
- add intrinsic undistortion
- add threaded capture
- attach a YOLO stage after `image_roi`
- add headless deploy mode

## Next sensible extension

The next clean step is to split deployment again into:
- `trigger_logic.*`
- `capture_writer.*`
- `overlay_debug.*`

Once you tell me what should happen with `image_roi`, that part can slot in without touching the calibration path too much.
