from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _stringify(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _stringify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_stringify(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def _safe_metrics(results: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for attr in ("top1", "top5", "fitness", "speed", "results_dict", "save_dir"):
        if hasattr(results, attr):
            out[attr] = _stringify(getattr(results, attr))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train Ultralytics classification model, validate, and export ONNX/NCNN."
    )
    parser.add_argument("--data", required=True, help="Path to classification dataset root")
    parser.add_argument("--model", default="yolo26n-cls.pt", help="Base model or checkpoint")
    parser.add_argument("--imgsz", type=int, default=100, help="Input image size")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--device", default="0", help="Device, e.g. 0 or cpu")
    parser.add_argument("--patience", type=int, default=8, help="Early-stop patience")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--project", default="runs/classify", help="Output project dir")
    parser.add_argument("--name", default="exp", help="Run name")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting existing run name")
    parser.add_argument("--test-split", default="test", choices=["train", "val", "test"], help="Split for held-out evaluation")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and use --model as checkpoint")
    parser.add_argument("--skip-export", action="store_true", help="Skip ONNX/NCNN export")
    parser.add_argument("--summary", default="train_export_summary.json", help="Path to summary JSON")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:  # pragma: no cover
        print("Failed to import ultralytics. Install with: pip install -U ultralytics[export]", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        return 2

    summary: dict[str, Any] = {
        "args": vars(args),
        "train": {},
        "val": {},
        "test": {},
        "exports": {},
        "artifacts": {},
    }

    best_pt_path: Path

    if args.skip_train:
        best_pt_path = Path(args.model).resolve()
        if not best_pt_path.exists():
            print(f"Checkpoint not found: {best_pt_path}", file=sys.stderr)
            return 2
        print(f"[skip-train] Using checkpoint: {best_pt_path}")
    else:
        print("[1/4] Training classification model...")
        model = YOLO(args.model)
        train_results = model.train(
            data=args.data,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            patience=args.patience,
            workers=args.workers,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
        )
        summary["train"] = _safe_metrics(train_results)

        save_dir = Path(getattr(train_results, "save_dir", Path(args.project) / args.name))
        best_pt_path = (save_dir / "weights" / "best.pt").resolve()
        if not best_pt_path.exists():
            print(f"best.pt not found after training: {best_pt_path}", file=sys.stderr)
            return 2
        summary["artifacts"]["best_pt"] = str(best_pt_path)
        print(f"[done] best.pt: {best_pt_path}")

    print("[2/4] Validating on val split...")
    best_model = YOLO(str(best_pt_path))
    val_results = best_model.val(data=args.data, imgsz=args.imgsz, split="val", device=args.device)
    summary["val"] = _safe_metrics(val_results)

    print(f"[3/4] Evaluating on {args.test_split} split...")
    test_results = best_model.val(data=args.data, imgsz=args.imgsz, split=args.test_split, device=args.device)
    summary[args.test_split] = _safe_metrics(test_results)

    if not args.skip_export:
        print("[4/4] Exporting ONNX...")
        onnx_out = best_model.export(format="onnx", imgsz=args.imgsz)
        summary["exports"]["onnx"] = _stringify(onnx_out)
        if isinstance(onnx_out, (str, Path)):
            summary["artifacts"]["onnx"] = str(onnx_out)

        print("[4/4] Exporting NCNN...")
        ncnn_out = best_model.export(format="ncnn", imgsz=args.imgsz)
        summary["exports"]["ncnn"] = _stringify(ncnn_out)
        if isinstance(ncnn_out, (str, Path)):
            summary["artifacts"]["ncnn"] = str(ncnn_out)

    summary_path = Path(args.summary).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_stringify(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Summary ===")
    print(f"best.pt: {best_pt_path}")
    if "top1" in summary.get("val", {}):
        print(f"val top1={summary['val'].get('top1')} top5={summary['val'].get('top5')}")
    if args.test_split in summary and "top1" in summary[args.test_split]:
        print(f"{args.test_split} top1={summary[args.test_split].get('top1')} top5={summary[args.test_split].get('top5')}")
    if "onnx" in summary.get("artifacts", {}):
        print(f"onnx: {summary['artifacts']['onnx']}")
    if "ncnn" in summary.get("artifacts", {}):
        print(f"ncnn: {summary['artifacts']['ncnn']}")
    print(f"summary json: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())