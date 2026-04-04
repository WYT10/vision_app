#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def stringify(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): stringify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [stringify(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def safe_metrics(results: Any) -> dict[str, Any]:
    return {k: stringify(getattr(results, k)) for k in ('top1', 'top5', 'fitness', 'speed', 'results_dict', 'save_dir') if hasattr(results, k)}


def parse_sizes(value: str | None, fallback: int) -> list[int]:
    if not value:
        return [fallback]
    out = [int(tok.strip()) for tok in value.split(',') if tok.strip()]
    if not out:
        raise argparse.ArgumentTypeError('sizes must not be empty')
    return out


def dataset_for(args: argparse.Namespace, size: int) -> Path:
    if args.data_template:
        return Path(args.data_template.format(size=size)).resolve()
    if args.data:
        return Path(args.data).resolve()
    raise SystemExit('Need --data or --data-template')


def export_formats(best_model: Any, imgsz: int, formats: list[str], summary: dict[str, Any]) -> None:
    for fmt in formats:
        print(f'[export] {fmt} imgsz={imgsz}')
        out = best_model.export(format=fmt, imgsz=imgsz)
        summary['exports'][fmt] = stringify(out)
        if isinstance(out, (str, Path)):
            summary['artifacts'][fmt] = str(out)


def train_one_size(args: argparse.Namespace, size: int, formats: list[str]) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except Exception as e:
        print('Failed to import ultralytics. Install with: pip install -U ultralytics[export]', file=sys.stderr)
        print(f'Import error: {e}', file=sys.stderr)
        raise

    data_root = dataset_for(args, size)
    run_name = f'{args.name}_px{size}' if len(parse_sizes(args.sizes, args.imgsz)) > 1 else args.name
    summary: dict[str, Any] = {
        'imgsz': size,
        'data': str(data_root),
        'train': {},
        'val': {},
        args.test_split: {},
        'exports': {},
        'artifacts': {},
    }

    if args.skip_train:
        best_pt_path = Path(args.model).resolve()
        if not best_pt_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {best_pt_path}')
        print(f'[skip-train] imgsz={size} checkpoint={best_pt_path}')
    else:
        print(f'[train] imgsz={size} data={data_root}')
        model = YOLO(args.model)
        train_results = model.train(
            data=str(data_root), imgsz=size, epochs=args.epochs, batch=args.batch, device=args.device,
            patience=args.patience, workers=args.workers, project=args.project, name=run_name, exist_ok=args.exist_ok,
        )
        summary['train'] = safe_metrics(train_results)
        save_dir = Path(getattr(train_results, 'save_dir', Path(args.project) / run_name))
        best_pt_path = (save_dir / 'weights' / 'best.pt').resolve()
        if not best_pt_path.exists():
            raise FileNotFoundError(f'best.pt not found after training: {best_pt_path}')
        summary['artifacts']['best_pt'] = str(best_pt_path)

    best_model = YOLO(str(best_pt_path))
    summary['val'] = safe_metrics(best_model.val(data=str(data_root), imgsz=size, split='val', device=args.device))
    summary[args.test_split] = safe_metrics(best_model.val(data=str(data_root), imgsz=size, split=args.test_split, device=args.device))
    if not args.skip_export:
        export_formats(best_model, size, formats, summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description='Train Ultralytics classification model(s), validate, and export ONNX + NCNN.')
    ap.add_argument('--data', default='', help='Single dataset root')
    ap.add_argument('--data-template', default='', help='Dataset template with {size}, e.g. merged/px{size}')
    ap.add_argument('--model', default='yolo26n-cls.pt', help='Base model or checkpoint')
    ap.add_argument('--imgsz', type=int, default=100, help='Single input size fallback')
    ap.add_argument('--sizes', default='', help='Comma-separated sizes, e.g. 16,40,128')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--device', default='0')
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--project', default='runs/classify')
    ap.add_argument('--name', default='exp')
    ap.add_argument('--exist-ok', action='store_true')
    ap.add_argument('--test-split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--skip-export', action='store_true')
    ap.add_argument('--export-formats', default='onnx,ncnn', help='Comma-separated export formats; default keeps laptop ONNX + NCNN flow')
    ap.add_argument('--summary', default='train_export_summary.json')
    args = ap.parse_args()

    sizes = parse_sizes(args.sizes, args.imgsz)
    formats = [tok.strip() for tok in args.export_formats.split(',') if tok.strip()]
    if args.skip_export:
        formats = []

    summary: dict[str, Any] = {'args': vars(args), 'runs': {}, 'recommended_deploy': None}
    best_key = None
    best_top1 = -1.0
    for size in sizes:
        run_summary = train_one_size(args, size, formats)
        key = f'px{size}'
        summary['runs'][key] = stringify(run_summary)
        try:
            top1 = float(run_summary.get('val', {}).get('top1', -1.0))
        except Exception:
            top1 = -1.0
        if top1 > best_top1:
            best_top1 = top1
            best_key = key
    summary['recommended_deploy'] = best_key

    summary_path = Path(args.summary).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(stringify(summary), indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'recommended_deploy': best_key, 'summary': str(summary_path)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
