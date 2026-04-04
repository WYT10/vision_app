
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

## Automation loop (laptop server + Pi collector)

### Laptop side

Run the controller server on your laptop. The dataset root should contain class subfolders of images.

```bash
python3 laptop_tools/controller_server.py \
  --dataset-root /path/to/display_images \
  --output-root /path/to/controller_runs \
  --port 8787
```

Open this on the iPad:

```text
http://<laptop-ip>:8787/display
```

### Pi side

Run deploy with automation enabled. The Pi will poll the laptop for the current trial, run the calibrated trigger/model path, and POST results back.

```bash
./vision_app --mode deploy \
  --config /home/pi/Desktop/vision_app/config/profile.conf \
  --automation-enable 1 \
  --automation-server-url http://<laptop-ip>:8787 \
  --automation-session demo \
  --automation-collect-dir /home/pi/Desktop/vision_app/report/automation
```

### Quick retrain on laptop

Assuming you already have a base Ultralytics classification dataset (`train/val/test`) and the training/export script from this repo:

```bash
python3 laptop_tools/quick_train.py \
  --base-dataset /path/to/base_dataset \
  --run-dir /path/to/controller_runs/session_YYYYMMDD_HHMMSS \
  --out-dataset /path/to/merged_dataset \
  --trainer /path/to/eval_export_cls.py \
  --model /path/to/current_best.pt \
  --imgsz 128 \
  --epochs 12 \
  --batch 64 \
  --device 0
```


## Automation training loop

### Laptop controller modes
- `demo`: iPad display + Pi prediction + result logging only. No ROI image saving, no retrain.
- `collect_retrain`: everything from demo, plus save wrong / low-confidence samples, then optionally trigger retrain after a threshold.

### Start laptop controller
```bash
python3 laptop_tools/controller_server.py \
  --dataset-root /path/to/display_images \
  --output-root /path/to/controller_runs \
  --mode demo
```

Open on iPad:
```
http://<laptop-ip>:8787/display
```

For collection + retrain:
```bash
python3 laptop_tools/controller_server.py \
  --dataset-root /path/to/display_images \
  --output-root /path/to/controller_runs \
  --mode collect_retrain \
  --low-conf-threshold 0.65 \
  --max-saved-per-class 200 \
  --max-saved-total 3000 \
  --disk-cap-mb 2048 \
  --retrain-threshold 300
```

### Pi deploy automation
```bash
./vision_app --mode deploy \
  --config /home/pi/Desktop/vision_app/config/profile.conf \
  --automation-enable 1 \
  --automation-mode demo \
  --automation-server-url http://<laptop-ip>:8787
```

Switch to collection mode:
```bash
./vision_app --mode deploy \
  --config /home/pi/Desktop/vision_app/config/profile.conf \
  --automation-enable 1 \
  --automation-mode collect_retrain \
  --automation-server-url http://<laptop-ip>:8787
```

### Synthetic + real retrain pipeline
Use `training_tools/live_tune_aug.py` to export `aug_config.json` for **synthetic data only**. Collected real hard examples are merged directly and are not passed through the live-tuner sequence.

Full pipeline:
```bash
python3 training_tools/run_retrain_pipeline.py \
  --src-dir /path/to/img_dataset \
  --aug-config /path/to/aug_config.json \
  --run-dir /path/to/controller_runs/session_xxx \
  --workspace-root /path/to/training_workspace \
  --base-model /path/to/yolo26n-cls.pt \
  --sizes 16,40,128 \
  --epochs 12 \
  --batch 64 \
  --device 0
```

Outputs are organized under `training_workspace/` into synthetic datasets, merged datasets, training runs, and multi-size summaries.


### Output paths and what they mean
- `controller_runs/session_xxx/results.jsonl`: one line per Pi result post.
- `controller_runs/session_xxx/hard_examples/<class>/`: wrong predictions kept for retraining.
- `controller_runs/session_xxx/low_confidence/<class>/`: correct but low-confidence examples, only if collection mode allows them.
- `training_workspace/synthetic/px16_synthetic/`, `px40_synthetic/`, `px128_synthetic/`: synthetic datasets generated from `img_dataset/` and optional `aug_config.json`.
- `training_workspace/merged/px16/`, `px40/`, `px128/`: synthetic datasets with real hard examples merged into `train/`.
- `training_workspace/runs/`: Ultralytics training runs.
- `training_workspace/reports/multi_size_summary.json`: combined 16/40/128 comparison and recommended deploy size.

### Direction split
- `demo` = iPad display + Pi prediction + result logging only. No ROI save, no retrain.
- `collect_retrain` = demo + ROI saving + threshold-based retrain request.
- `live_tune_aug.py` only affects synthetic generation through `aug_config.json`. It does **not** re-transform collected real hard examples.


## Pulling updates and overwriting local changes

If you want to ignore local changes and let the remote branch overwrite your local tracked files, including `config/profile.conf`, use:

```bash
cd ~/Desktop/vision_app
git fetch https://github.com/WYT10/vision_app automation
git reset --hard FETCH_HEAD
```

This will:
- discard local edits in tracked files
- move your working tree to the pulled `automation` branch tip
- overwrite local config changes

If you also want to remove untracked local files and folders:

```bash
cd ~/Desktop/vision_app
git fetch https://github.com/WYT10/vision_app automation
git reset --hard FETCH_HEAD
git clean -fd
```

Use `git clean -fd` carefully. It deletes untracked files and directories.

### Recommended overwrite-all update flow

```bash
cd ~/Desktop/vision_app
git fetch https://github.com/WYT10/vision_app automation
git reset --hard FETCH_HEAD

export CMAKE_PREFIX_PATH=/home/pi/ncnn/build/install:$CMAKE_PREFIX_PATH
rm -rf build
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

### If you only want to overwrite `config/profile.conf`

```bash
cd ~/Desktop/vision_app
git checkout -- config/profile.conf
git pull https://github.com/WYT10/vision_app automation
```
