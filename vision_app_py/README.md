# vision_app_py

Single-file Python version of the vision app with similar ideas/arguments:

- `--mode probe`
- `--mode calibrate`
- `--mode deploy`

It uses:
- USB camera via OpenCV
- AprilTag detection through `cv2.aruco` AprilTag dictionaries
- ONNX Runtime for ONNX
- optional Python `ncnn` module for NCNN

## Install

```bash
pip install opencv-python numpy onnxruntime
```

Optional NCNN Python support:

```bash
pip install ncnn
```

## Probe

```bash
python vision_app_py.py --mode probe --device /dev/video0 --width 160 --height 120 --fps 120 --fourcc MJPG --duration 3 --preview 1 --report probe.json
```

Or scan common modes:

```bash
python vision_app_py.py --mode probe --device /dev/video0 --scan-common 1 --duration 2 --preview 0 --report probe.json
```

## Calibrate

```bash
python vision_app_py.py --mode calibrate \
  --device /dev/video0 \
  --width 160 --height 120 --fps 120 --fourcc MJPG \
  --tag-family auto --target-id 0 \
  --warp-width 384 --warp-height 384 \
  --target-tag-px 128 \
  --save-config ../report/vision_app_calibration.json
```

Controls:
- SPACE / ENTER: lock current tag
- u: unlock
- 1 / 2: select red_roi / image_roi
- w a s d: move ROI
- i/k: height -/+
- j/l: width -/+
- [ / ]: move step -/+
- , / .: size step -/+
- p: save
- q / ESC: quit

## Deploy with ONNX

```bash
python vision_app_py.py --mode deploy \
  --device /dev/video0 \
  --width 160 --height 120 --fps 30 --fourcc MJPG \
  --load-config ../report/vision_app_calibration.json \
  --model-enable 1 --model-backend onnx \
  --model-onnx-path ../models/best.onnx \
  --model-labels-path ../models/labels.txt \
  --model-input-width 128 --model-input-height 128 \
  --model-preprocess crop \
  --model-threads 2 --model-stride 1 --model-max-hz 2
```

## Deploy with NCNN

```bash
python vision_app_py.py --mode deploy \
  --device /dev/video0 \
  --width 160 --height 120 --fps 30 --fourcc MJPG \
  --load-config ../report/vision_app_calibration.json \
  --model-enable 1 --model-backend ncnn \
  --model-ncnn-param-path ../models/model.ncnn.param \
  --model-ncnn-bin-path ../models/model.ncnn.bin \
  --model-labels-path ../models/labels.txt \
  --model-input-width 128 --model-input-height 128 \
  --model-preprocess crop \
  --model-threads 4 --model-stride 1 --model-max-hz 5
```

## Partial pipeline tests

Red ROI only:

```bash
python vision_app_py.py --mode deploy ... --run-red 1 --run-image-roi 0 --run-model 0
```

Image ROI capture only:

```bash
python vision_app_py.py --mode deploy ... --run-red 0 --run-image-roi 1 --run-model 0 --save-image-roi-dir ./roi_dump --save-every-n 10
```

Model only on live ROI:

```bash
python vision_app_py.py --mode deploy ... --run-red 0 --run-image-roi 1 --run-model 1
```

## Notes

- Start gentle on Pi:
  - `--fps 30`
  - `--headless 1`
  - `--model-max-hz 2` for ONNX
  - `--model-max-hz 5` for NCNN
- If deploy looks wrong, make sure live camera mode matches the saved calibration source size.
