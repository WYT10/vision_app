
# vision_app

# Drop-in replacement notes

Replace your project files with **this entire drop-in set** (do not mix old/new headers).
At minimum, replace:
- `src/`
- `cmake/`
- `CMakeLists.txt`
- `README.md`
- `vision_app.conf`

This avoids the mixed old/new API state that caused the previous build failures.

Ready-to-run drop-in build for:
- `probe`
- `calibrate`
- `deploy`

This version keeps:
- camera probing and FPS benchmarking helpers
- AprilTag calibration UI and ROI editing
- ONNX Runtime classifier backend
- NCNN classifier backend
- separate testing toggles for `red_roi`, `image_roi`, and model inference
- prediction rate limiting with `--model-max-hz`

## Pull latest project

```bash
git pull https://github.com/WYT10/vision_app.git main
```

## Build on Raspberry Pi

```bash
export CMAKE_PREFIX_PATH=/home/pi/ncnn/build/install:$CMAKE_PREFIX_PATH

cd ~/Desktop/vision_app
rm -rf build
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

## Modes

### 1) Probe
Show supported camera modes from `v4l2-ctl`:

```bash
./vision_app --mode probe --device /dev/video0
```

### 2) Calibrate
Open the camera, detect the AprilTag, and map the 4 detected tag corners to a centered
`target_tag_px x target_tag_px` square inside the warped canvas `warp_width x warp_height`.

Use the UI to:
- lock/unlock
- move/resize `red_roi`
- move/resize `image_roi`
- save warp / save rois / save all

Example:

```bash
./vision_app \
  --mode calibrate \
  --device /dev/video0 \
  --width 160 --height 120 --fourcc MJPG --fps 120 \
  --warp-width 384 --warp-height 384 \
  --target-tag-px 128 \
  --tag-family auto --target-id 0 \
  --save-warp ../report/warp_package.yml.gz \
  --save-rois ../report/rois.yml
```

Controls:
- `SPACE` / `ENTER`: lock current tag
- `u`: unlock
- `1` / `2`: select `red_roi` / `image_roi`
- `w a s d`: move ROI
- `i / k`: height - / +
- `j / l`: width - / +
- `[` / `]`: move step down / up
- `,` / `.`: size step down / up
- `p`: save all
- `y`: save warp only
- `o`: save rois only
- `r`: reset rois
- `q` / `ESC`: quit

### 3) Deploy
Load saved warp + rois, compute `red_roi`, crop `image_roi`, and optionally classify with ONNX or NCNN.

#### NCNN deploy example

```bash
./vision_app \
  --mode deploy \
  --device /dev/video0 \
  --width 160 --height 120 --fourcc MJPG --fps 120 \
  --load-warp ../report/warp_package.yml.gz \
  --load-rois ../report/rois.yml \
  --model-enable 1 \
  --model-backend ncnn \
  --model-ncnn-param-path ../models/model.ncnn.param \
  --model-ncnn-bin-path ../models/model.ncnn.bin \
  --model-labels-path ../models/labels.txt \
  --model-input-width 128 --model-input-height 128 \
  --model-preprocess crop \
  --model-threads 4 \
  --model-stride 1 \
  --model-max-hz 5.0
```

#### ONNX deploy example

```bash
./vision_app \
  --mode deploy \
  --device /dev/video0 \
  --width 160 --height 120 --fourcc MJPG --fps 120 \
  --load-warp ../report/warp_package.yml.gz \
  --load-rois ../report/rois.yml \
  --model-enable 1 \
  --model-backend onnx \
  --model-onnx-path ../models/best.onnx \
  --model-labels-path ../models/labels.txt \
  --model-input-width 128 --model-input-height 128 \
  --model-preprocess crop \
  --model-threads 4 \
  --model-stride 1 \
  --model-max-hz 5.0
```

## Separate testing in deploy

You can test the three subsystems separately:

### Red ROI only
```bash
./vision_app --mode deploy ... --run-red 1 --run-image-roi 0 --run-model 0
```

### Image ROI capture only
```bash
./vision_app --mode deploy ... --run-red 0 --run-image-roi 1 --run-model 0 \
  --save-image-roi-dir ../captures/image_roi --save-every-n 10
```

### Full classification
```bash
./vision_app --mode deploy ... --run-red 1 --run-image-roi 1 --run-model 1
```

## Notes

- Keep the **same camera mode** between calibration and deploy.
- The saved warp package stores the original calibration source size. If deploy uses a different camera size,
  the app prints a warning because the remap no longer matches the original geometry.
- `image_roi` should ideally match the model input size used during training.
- Use NCNN as the default deploy backend on Pi; keep ONNX for parity/debug.

## Example config (`vision_app.conf`)

```ini
mode=deploy
device=/dev/video0
width=160
height=120
fps=120
fourcc=MJPG
ui=1

warp_width=384
warp_height=384
target_tag_px=128
tag_family=auto
target_id=0
manual_lock_only=1
lock_frames=4

save_warp=../report/warp_package.yml.gz
load_warp=../report/warp_package.yml.gz
save_rois=../report/rois.yml
load_rois=../report/rois.yml
save_report=../report/latest_report.md

red_roi=0.08,0.08,0.18,0.18
image_roi=0.32,0.10,0.50,0.55

red_h1_low=0
red_h1_high=10
red_h2_low=170
red_h2_high=180
red_s_min=80
red_v_min=60

model_enable=1
model_backend=ncnn
model_ncnn_param_path=../models/model.ncnn.param
model_ncnn_bin_path=../models/model.ncnn.bin
model_labels_path=../models/labels.txt
model_input_width=128
model_input_height=128
model_preprocess=crop
model_threads=4
model_stride=1
model_max_hz=5.0
model_topk=5

run_red=1
run_image_roi=1
run_model=1
save_every_n=0
```
