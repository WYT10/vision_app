# Portable Classifier Inference Bundle v4

This bundle gives you one C++ executable that can run either:
- ONNX Runtime (`--backend onnx`)
- NCNN (`--backend ncnn`)

It also adds benchmarking helpers so you can compare:
- top-1 accuracy on `train/`, `val/`, `test/`
- end-to-end throughput (`img/s`, `ms/img`)
- repeated single-image latency without disk I/O noise

## Folder layout

- `src/` — C++ source
- `cmake/` — `FindONNXRuntime.cmake`
- `tools/compare_backends.py` — helper to benchmark ONNX vs NCNN over dataset splits

## Build

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

If NCNN is not globally discoverable, set either:
- `CMAKE_PREFIX_PATH=/home/pi/ncnn/build/install`
- or `ncnn_DIR=/home/pi/ncnn/build/install/lib/cmake/ncnn`

## Single image run

### ONNX

```bash
./portable_cls_infer \
  --backend onnx \
  --model /path/to/best.onnx \
  --labels /path/to/labels.txt \
  --input /path/to/test.jpg \
  --prep crop
```

### NCNN

```bash
./portable_cls_infer \
  --backend ncnn \
  --model /path/to/model.ncnn.param \
  --weights /path/to/model.ncnn.bin \
  --labels /path/to/labels.txt \
  --input /path/to/test.jpg \
  --prep crop \
  --threads 4
```

## Dataset evaluation

This compares predictions against the parent folder label.

```bash
./portable_cls_infer \
  --backend ncnn \
  --model /path/to/model.ncnn.param \
  --weights /path/to/model.ncnn.bin \
  --labels /path/to/labels.txt \
  --input /path/to/dataset_cls/test \
  --prep crop \
  --threads 4 \
  --eval-parent-label \
  --quiet-per-image \
  --summary-json ./ncnn_test_summary.json \
  --per-class-csv ./ncnn_test_per_class.csv
```

## Repeated latency test

Use this to separate runtime speed from disk/image loading.

```bash
./portable_cls_infer \
  --backend ncnn \
  --model /path/to/model.ncnn.param \
  --weights /path/to/model.ncnn.bin \
  --labels /path/to/labels.txt \
  --input /path/to/test.jpg \
  --prep crop \
  --threads 4 \
  --repeat 200 \
  --warmup 20 \
  --quiet-per-image \
  --summary-json ./ncnn_latency.json
```

## Compare ONNX vs NCNN across splits

```bash
python3 tools/compare_backends.py \
  --exe ./build/portable_cls_infer \
  --onnx-model /home/pi/Desktop/vision_app/models/best.onnx \
  --ncnn-param /home/pi/Desktop/vision_app/models/model.ncnn.param \
  --ncnn-bin /home/pi/Desktop/vision_app/models/model.ncnn.bin \
  --labels /home/pi/Desktop/vision_app/models/labels.txt \
  --dataset-root /home/pi/Desktop/vision_app/2026_SmartCar_Loongson_Object_Recognition-main/2026_SmartCar_Loongson_Object_Recognition-main/dataset_cls \
  --output-dir ./bench_out \
  --split val \
  --split test \
  --prep crop \
  --threads 4 \
  --latency-image /home/pi/Desktop/vision_app/2026_SmartCar_Loongson_Object_Recognition-main/2026_SmartCar_Loongson_Object_Recognition-main/dataset_cls/test/I_body_armor/I_body_armor_00013.jpg
```

Outputs:
- `bench_out/compare_summary.csv`
- `bench_out/compare_summary.md`
- per-backend per-split JSON/CSV summaries

## Notes

- If ONNX and NCNN are similar on a single image, that is normal for a small classifier at `224x224` on a Pi 5.
- Dataset mode includes image loading and preprocessing, so pure runtime differences can look smaller than you expect.
- The repeated single-image latency mode is a better backend speed comparison.
