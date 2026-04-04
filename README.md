# vision_app

A cleaned vision stack with a **Pi NCNN-only runtime** and a **laptop-side Python training/export pipeline**.

## What runs where

### Pi runtime (C++)
- `probe`: camera test / capability check
- `live`: raw preview or headless FPS test
- `calibrate`: AprilTag lock + warp save + ROI tuning
- `deploy`: load saved profile/warp and run trigger + NCNN classifier
- optional automation client inside `deploy`

### Laptop tools (Python)
- `controller_server.py`: truth controller + iPad display + result collection
- `prepare_cls_dataset.py`: synthetic dataset generation
- `merge_hard_examples.py`: merge collected camera ROI samples into the train split
- `eval_export_cls.py`: train from `.pt`, then export **ONNX + NCNN**
- `run_retrain_pipeline.py`: one-shot synthetic + merge + multi-size train/export

The Pi build and runtime do **not** require ONNX Runtime.
The laptop training/export pipeline still supports `.pt -> onnx -> ncnn`.

## Project layout

```text
vision_app/
├── CMakeLists.txt
├── README.md
├── config/
│   ├── profile.conf
│   └── profile.example.conf
├── laptop_tools/
│   ├── controller_server.py
│   └── quick_train.py
├── training_tools/
│   ├── eval_export_cls.py
│   ├── live_tune_aug.py
│   ├── merge_hard_examples.py
│   ├── prepare_cls_dataset.py
│   └── run_retrain_pipeline.py
└── src/
    ├── app_config.*
    ├── app_types.hpp
    ├── automation.hpp
    ├── calibrate.hpp
    ├── camera.*
    ├── classifier_common.hpp
    ├── deploy_runtime.hpp
    ├── main.cpp
    ├── model.*
    ├── ncnn_classifier.hpp
    ├── roi_helper.cpp
    ├── stats.hpp
    ├── status_ui.*
    └── trigger.*
```

## Pi build (NCNN only)

```bash
export CMAKE_PREFIX_PATH=/home/pi/ncnn/build/install:$CMAKE_PREFIX_PATH

cd ~/Desktop/vision_app
rm -rf build
mkdir -p build
cd build
cmake .. -Dncnn_DIR=/home/pi/ncnn/build/install/lib/cmake/ncnn
make -j1
```

Use `-j1` on the Pi if you want the lowest RAM and thermal load.

## Key runtime files

- `config/profile.conf`: editable runtime profile
- `report/warp_package.yml.gz`: saved homography artifact
- `report/rois.yml`: saved ROI geometry

## Runtime model backend

Pi runtime supports:
- `model_backend=ncnn`
- `model_backend=off`

There is no Pi-side ONNX Runtime dependency in this bundle.

## Helpful Pi commands

```bash
./vision_app --help
./vision_app --mode probe --device /dev/video0
./vision_app --mode calibrate --config /home/pi/Desktop/vision_app/config/profile.conf
./vision_app --mode deploy --config /home/pi/Desktop/vision_app/config/profile.conf
```

## Automation server path policy

The laptop controller treats:
- `--dataset-root` as **read-only input**
- `--output-root` as the **only writable root**

It creates:

```text
output-root/
  sessions/
    session_YYYYMMDD_HHMMSS/
  workspaces/
    session_YYYYMMDD_HHMMSS/
```

### `sessions/...`
Contains runtime collection output:
- `session.json`
- `results.jsonl`
- `hard_examples/`
- `low_confidence/`
- `correct/`
- `rejected/`

### `workspaces/...`
Contains training artifacts:
- `synthetic/`
- `merged/`
- `runs/`
- `reports/`

Nothing generated should need to spill outside `output-root`.

## Laptop controller

```bash
python3 laptop_tools/controller_server.py   --dataset-root /path/to/display_images   --output-root /path/to/automation   --mode demo
```

Open on the iPad:

```text
http://<laptop-ip>:8787/display
```

### Controller modes
- `demo`: show image + receive result + log only
- `collect_retrain`: also save wrong / low-confidence camera ROI samples

### Pi automation example

```bash
./vision_app --mode deploy   --config /home/pi/Desktop/vision_app/config/profile.conf   --automation-enable 1   --automation-mode demo   --automation-server-url http://<laptop-ip>:8787
```

## Training pipeline on laptop

### 1) Tune synthetic augmentation (optional)

```bash
python3 live_tune_aug.py
```

This exports `aug_config.json` for **synthetic dataset generation only**.
Collected real camera ROI images are **not** run through this tuner.

### 2) Full retrain pipeline

```bash
python3 training_tools/run_retrain_pipeline.py   --src-dir /path/to/img_dataset   --run-dir /path/to/automation/sessions/session_xxx   --workspace-root /path/to/automation/workspaces/session_xxx   --base-model /path/to/yolo26n-cls.pt   --sizes 16,40,128   --epochs 12   --batch 64   --device 0
```

This will:
1. generate synthetic datasets for each size
2. merge collected hard examples into `train/`
3. train each size from `.pt`
4. export **ONNX + NCNN**
5. write `multi_size_summary.json`

## Dataset split logic

For each size, the merged dataset is:
- `train` = synthetic train + collected hard examples
- `val` = synthetic val
- `test` = synthetic test

So the actual datasets used by the trainer live under:

```text
output-root/workspaces/session_xxx/merged/px16/
output-root/workspaces/session_xxx/merged/px40/
output-root/workspaces/session_xxx/merged/px128/
```

## Pulling updates and discarding local changes

Overwrite tracked local changes completely:

```bash
cd ~/Desktop/vision_app
git fetch https://github.com/WYT10/vision_app automation
git reset --hard FETCH_HEAD
```

Also remove untracked files/folders:

```bash
cd ~/Desktop/vision_app
git fetch https://github.com/WYT10/vision_app automation
git reset --hard FETCH_HEAD
git clean -fd
```
