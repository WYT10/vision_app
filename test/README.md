# Portable classifier benchmark helpers

These scripts work with the **older** `portable_cls_infer` binary that only supports:
- `--backend`
- `--model`
- `--weights`
- `--labels`
- `--input`
- `--prep`
- `--threads`
- `--topk`

They run the binary many times and produce detailed reports:
- `*_summary.json`
- `*_summary.md`
- `*_predictions.csv`
- `*_per_class.csv`
- `*_confusion_matrix.csv`
- `compare_summary.csv`
- `compare_summary.md`

## 1) Evaluate one backend on a whole split

```bash
python3 eval_classifier_cli.py \
  --exe ./build/portable_cls_infer \
  --backend ncnn \
  --model /home/pi/Desktop/vision_app/models/model.ncnn.param \
  --weights /home/pi/Desktop/vision_app/models/model.ncnn.bin \
  --labels /home/pi/Desktop/vision_app/models/labels.txt \
  --dataset-root /home/pi/Desktop/vision_app/2026_SmartCar_Loongson_Object_Recognition-main/2026_SmartCar_Loongson_Object_Recognition-main/dataset_cls \
  --split test \
  --prep crop \
  --threads 4 \
  --output-dir ./bench_out
```

For ONNX, just swap backend/model and remove `--weights`.

## 2) Repeat benchmark on one image

```bash
python3 eval_classifier_cli.py \
  --exe ./build/portable_cls_infer \
  --backend onnx \
  --model /home/pi/Desktop/vision_app/models/best.onnx \
  --labels /home/pi/Desktop/vision_app/models/labels.txt \
  --repeat-image /home/pi/Desktop/vision_app/2026_SmartCar_Loongson_Object_Recognition-main/2026_SmartCar_Loongson_Object_Recognition-main/dataset_cls/test/I_body_armor/I_body_armor_00013.jpg \
  --repeat 200 \
  --warmup 20 \
  --prep crop \
  --output-dir ./bench_out
```

## 3) Compare ONNX and NCNN directly

```bash
python3 compare_backends_cli.py \
  --exe ./build/portable_cls_infer \
  --eval-script ./eval_classifier_cli.py \
  --onnx-model /home/pi/Desktop/vision_app/models/best.onnx \
  --ncnn-param /home/pi/Desktop/vision_app/models/model.ncnn.param \
  --ncnn-bin /home/pi/Desktop/vision_app/models/model.ncnn.bin \
  --labels /home/pi/Desktop/vision_app/models/labels.txt \
  --dataset-root /home/pi/Desktop/vision_app/2026_SmartCar_Loongson_Object_Recognition-main/2026_SmartCar_Loongson_Object_Recognition-main/dataset_cls \
  --split val \
  --split test \
  --prep crop \
  --threads 4 \
  --latency-image /home/pi/Desktop/vision_app/2026_SmartCar_Loongson_Object_Recognition-main/2026_SmartCar_Loongson_Object_Recognition-main/dataset_cls/test/I_body_armor/I_body_armor_00013.jpg \
  --output-dir ./bench_compare
```

## Notes

- Classification uses the folder structure directly: `dataset_cls/train`, `dataset_cls/val`, `dataset_cls/test`.
- Ground truth is taken from the **parent folder name** of each image.
- This is equivalent to using the split folders in a YOLO-style classification dataset, but it does **not** need a YAML file.
