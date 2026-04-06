# vision_app

A calibrated Pi-side vision app plus laptop-side automation/training tools.

## Architecture

### Pi side
Pi runtime is **NCNN-only**.

Use it for:
- `probe`
- `live`
- `calibrate`
- `deploy`

Pi deploy can also run the automation loop by talking to the laptop server.

### Laptop side
Laptop Python tools do:
- iPad stimulus serving
- result collection
- hard-example saving
- dataset merge
- retraining
- `.pt -> onnx -> ncnn` export

## Folder layout

```text
vision_app/
  CMakeLists.txt
  README.md
  cmake/
  config/
  src/
  laptop_tools/
    controller_server.py
    quick_train.py
  training_tools/
    prepare_cls_dataset.py
    merge_session_into_dataset.py
    eval_export_cls.py
    live_tune_aug.py
    run_retrain_pipeline.py
  models/
  report/
```

There should be only **one copy** of each Python tool:
- server tools under `laptop_tools/`
- training/data tools under `training_tools/`

## Build on Pi

Use this as the normal rebuild command:

```bash
export CMAKE_PREFIX_PATH=/home/pi/ncnn/build/install:$CMAKE_PREFIX_PATH

cd ~/Desktop/vision_app
rm -rf build
mkdir -p build
cd build
cmake .. -Dncnn_DIR=/home/pi/ncnn/build/install/lib/cmake/ncnn
make -j1
```

Notes:
- Pi runtime is NCNN-only.
- ONNX Runtime is **not required** on the Pi.
- `make -j1` is recommended on the Pi for lower memory usage.

## Git commit / gitignore / force update

### Force update and overwrite all local changes

If you want to overwrite all local tracked changes and update to the latest branch state:

```bash
cd ~/Desktop/vision_app
git fetch https://github.com/WYT10/vision_app automation
git reset --hard FETCH_HEAD
```

If you also want to remove untracked files and folders:

```bash
cd ~/Desktop/vision_app
git fetch https://github.com/WYT10/vision_app automation
git reset --hard FETCH_HEAD
git clean -fd
```

### Important `.gitignore` note

If a file is already tracked, `.gitignore` does **not** stop Git from showing it.

`.gitignore` only ignores **untracked** files.

### Stop tracking a file that is now ignored

```bash
git rm --cached path/to/file
git commit -m "stop tracking generated/local files and honor gitignore"
```

### Stop tracking a folder that is now ignored

```bash
git rm -r --cached path/to/folder
git commit -m "stop tracking generated/local files and honor gitignore"
```

### Refresh the whole repo to match `.gitignore`

Use this only if you want Git to re-check the whole repo against `.gitignore`:

```bash
git rm -r --cached .
git add .
git commit -m "refresh tracked files to match gitignore"
```

Be careful with this command because it reindexes the whole repo.

### Practical rule

- tracked file + added to `.gitignore` = still tracked
- to really ignore it = `git rm --cached ...`
## Server pages

Run the laptop server with only these args:
- `--dataset-root`
- `--output-root`
- `--mode`
- `--port`

### Start the server

```powershell
python .\laptop_tools\controller_server.py `
  --dataset-root C:\path\to\img_dataset `
  --output-root C:\path\to\automation `
  --mode demo `
  --port 8787
```

### Pages
- demo page: `http://<laptop-ip>:8787/display`
- calibration tag page: `http://<laptop-ip>:8787/calibrate_tag`
- status API: `http://<laptop-ip>:8787/api/status`

### Stimulus format
Both `/display` and `/calibrate_tag` use the same layout:
- top image/tag: **240 x 240**
- white gap: **20 px**
- red rectangle: **240 x 100**

## Server modes

### `demo`
Does:
- serve the iPad pages
- log results
- advance to the next image after a successful Pi result post

Does **not**:
- save ROI images
- prepare retraining data

### `collect_retrain`
Does:
- everything from `demo`
- save camera ROI images when:
  - prediction is wrong
  - or confidence is low

Saved folders:
```text
output-root/
  sessions/
    session_xxx/
      results.jsonl
      hard_examples/
      low_confidence/
```

## Calibration flow

### 1. Open the calibration tag page on the iPad
```text
http://<laptop-ip>:8787/calibrate_tag
```

### 2. On the Pi, run calibrate and save
Use your normal calibrate command, then:
- lock the tag
- tune ROI/trigger
- press `p` to save

### 3. Switch the iPad to the demo page
```text
http://<laptop-ip>:8787/display
```

After that, use deploy for runtime automation.

## Pi deploy: normal NCNN inference

Example for a 40 px model:

```bash
./vision_app \
  --mode deploy \
  --device rtsp://192.168.0.115:5500/camera \
  --width 320 --height 240 --fps 25 \
  --config /home/pi/Desktop/vision_app/config/profile.conf \
  --trigger-mode dynamic_red_stacked \
  --model-enable 1 \
  --model-backend ncnn \
  --model-ncnn-param-path /home/pi/Desktop/vision_app/models/40_2/model.ncnn.param \
  --model-ncnn-bin-path /home/pi/Desktop/vision_app/models/40_2/model.ncnn.bin \
  --model-labels-path /home/pi/Desktop/vision_app/models/40_2/labels.txt \
  --model-input-width 40 \
  --model-input-height 40
```

### Important
Do **not** confuse:
- `--model-input-width/height 40` = classifier input size
- `--target-tag-px 40` = warp geometry scale

These are different.

In general, keep the calibrated geometry from `profile.conf` unless you explicitly recalibrate.

## Pi deploy: automation demo

Use this when you want the iPad image to advance but **do not** want to save training images.

```bash
./vision_app \
  --mode deploy \
  --device rtsp://192.168.0.115:5500/camera \
  --width 320 --height 240 --fps 25 \
  --config /home/pi/Desktop/vision_app/config/profile.conf \
  --trigger-mode dynamic_red_stacked \
  --model-enable 1 \
  --model-backend ncnn \
  --model-ncnn-param-path /home/pi/Desktop/vision_app/models/40_2/model.ncnn.param \
  --model-ncnn-bin-path /home/pi/Desktop/vision_app/models/40_2/model.ncnn.bin \
  --model-labels-path /home/pi/Desktop/vision_app/models/40_2/labels.txt \
  --model-input-width 40 \
  --model-input-height 40 \
  --automation-enable 1 \
  --automation-mode demo \
  --automation-server-url http://192.168.0.153:8787 \
  --automation-session demo \
  --automation-poll-ms 250
```

## Pi deploy: automation collect + retrain

Use this when you want to collect wrong or low-confidence camera ROI images.

```bash
./vision_app \
  --mode deploy \
  --device rtsp://192.168.0.115:5500/camera \
  --width 320 --height 240 --fps 25 \
  --config /home/pi/Desktop/vision_app/config/profile.conf \
  --trigger-mode dynamic_red_stacked \
  --model-enable 1 \
  --model-backend ncnn \
  --model-ncnn-param-path /home/pi/Desktop/vision_app/models/40_2/model.ncnn.param \
  --model-ncnn-bin-path /home/pi/Desktop/vision_app/models/40_2/model.ncnn.bin \
  --model-labels-path /home/pi/Desktop/vision_app/models/40_2/labels.txt \
  --model-input-width 40 \
  --model-input-height 40 \
  --automation-enable 1 \
  --automation-mode collect_retrain \
  --automation-server-url http://192.168.0.153:8787 \
  --automation-session collect_retrain \
  --automation-poll-ms 250 \
  --automation-collect-dir /home/pi/Desktop/vision_app/report/automation
```

### Required condition for image saving
To save training images, **both** must be true:
- laptop server started with `--mode collect_retrain`
- Pi started with `--automation-mode collect_retrain`

## What gets saved

Saved training images are the **camera ROI images**, not the original dataset image.

Why:
- retraining should use what the camera actually sees
- this keeps train input aligned with deploy input

## Label matching

The server supports bilingual label matching.

Examples:
- `A-枪支` matches `A_gun`
- `B-爆炸物` matches `B_explosive`

The server:
- recomputes `match` on the server side
- supports built-in bilingual aliases
- also supports optional override files:
  - `output-root/label_aliases.json`
  - `dataset-root/label_aliases.json`

Example alias file:

```json
{
  "A-枪支": "A_gun",
  "A_gun": "A_gun",
  "B-爆炸物": "B_explosive",
  "B_explosive": "B_explosive"
}
```

## Existing collected data: what to do next

If you already have a session folder like:

```text
automation/
  sessions/
    session_20260406_162533/
      results.jsonl
      hard_examples/
      low_confidence/
```

then you can retrain immediately.

You do **not** need to recollect first.

## Base dataset vs new ROI images

There are two different image sources:

### 1. Original dataset
This needs preprocessing / synthetic generation first.

Use:
```text
training_tools/prepare_cls_dataset.py
```

This creates a generated classification dataset with:
- `train/`
- `val/`
- `test/`
- `labels.txt`
- `class_to_idx.json`
- `dataset_summary.json`

### 2. Collected ROI images
These are already real camera observations.

Do **not** run them through the aggressive synthetic preprocessing pipeline again.

Instead:
- merge them directly into `train/<class>/`
- keep `val/` and `test/` unchanged

## Merge script

Use this script:

```text
training_tools/merge_session_into_dataset.py
```

It takes:
- the generated base dataset
- the session folder

and creates one final merged dataset folder.

### Merge command

```powershell
python .\training_tools\merge_session_into_dataset.py `
  --base-dataset C:\path\to\generated_base_40_dataset `
  --session-dir C:\path\to\automation\sessions\session_20260406_162533 `
  --out-dataset C:\path\to\automation\workspaces\session_20260406_162533\merged\px40 `
  --overwrite
```

What it does:
- copies the base dataset
- merges `hard_examples/` into `train/`
- merges `low_confidence/` into `train/` by default
- keeps `val/` and `test/` unchanged
- resizes the merged ROI images to the base dataset target size if available

### Result
You get one final dataset folder like:

```text
automation/workspaces/session_20260406_162533/merged/px40/
  train/
  val/
  test/
  labels.txt
  class_to_idx.json
  dataset_summary.json
  merge_summary.json
```

This is the single dataset folder you pass to training.

## Retrain from the previous model

Use:
```text
training_tools/eval_export_cls.py
```

### Important
Train from the previous **`best.pt`**, not from NCNN.

Use:
- previous `best.pt` = training checkpoint
- new exported `.ncnn.param` + `.ncnn.bin` = Pi deploy artifacts

### Retrain command for 40 px

```powershell
python .\training_tools\eval_export_cls.py `
  --data C:\path\to\automation\workspaces\session_20260406_162533\merged\px40 `
  --model C:\path\to\previous\best.pt `
  --imgsz 40 `
  --epochs 12 `
  --batch 64 `
  --device 0 `
  --project C:\path\to\automation\workspaces\session_20260406_162533\runs `
  --name px40_retrain `
  --exist-ok `
  --summary C:\path\to\automation\workspaces\session_20260406_162533\runs\px40_retrain\summary.json
```

This will:
- continue from the previous checkpoint
- train on the merged dataset
- validate and evaluate
- export **ONNX**
- export **NCNN**

## Export only from an existing checkpoint

If you only want export and evaluation from an existing `best.pt`:

```powershell
python .\training_tools\eval_export_cls.py `
  --data C:\path\to\automation\workspaces\session_20260406_162533\merged\px40 `
  --model C:\path\to\previous\best.pt `
  --imgsz 40 `
  --device 0 `
  --project C:\path\to\automation\workspaces\session_20260406_162533\runs `
  --name px40_export_only `
  --exist-ok `
  --skip-train `
  --summary C:\path\to\automation\workspaces\session_20260406_162533\runs\px40_export_only\summary.json
```

## Debug checklist

### If the iPad image does not advance
Check:
- server is running
- Pi is started with `--automation-enable 1`
- Pi can reach `--automation-server-url`
- `results.jsonl` is growing
- `http://<laptop-ip>:8787/api/status` shows `last_post_summary` updating

### If no training images are saved
Check:
- server is in `collect_retrain`
- Pi is in `collect_retrain`
- `roi_jpg_b64` is non-empty in `results.jsonl`
- the prediction is actually wrong or low-confidence

### If Git still shows ignored files
See the Git section above. `.gitignore` does not untrack already-tracked files.

## Recommended workflow

### First pass
1. `demo`
2. verify image advances
3. verify labels are correct

### Second pass
1. `collect_retrain`
2. collect ROI images

### Third pass
1. merge the session into the base dataset
2. retrain from the previous `best.pt`
3. export new ONNX + NCNN
4. deploy the new NCNN back to the Pi
