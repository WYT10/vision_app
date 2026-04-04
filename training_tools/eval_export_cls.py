#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


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
    for attr in ('top1', 'top5', 'fitness', 'speed', 'results_dict', 'save_dir'):
        if hasattr(results, attr):
            out[attr] = _stringify(getattr(results, attr))
    return out


def train_one_size(data_root: Path,
                   model_ref: str,
                   imgsz: int,
                   epochs: int,
                   batch: int,
                   device: str,
                   patience: int,
                   workers: int,
                   project: str,
                   name: str,
                   exist_ok: bool,
                   test_split: str,
                   skip_train: bool,
                   skip_export: bool) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except Exception as e:  # pragma: no cover
        print('Failed to import ultralytics. Install with: pip install -U ultralytics[export]', file=sys.stderr)
        print(f'Import error: {e}', file=sys.stderr)
        raise

    summary: dict[str, Any] = {
        'imgsz': imgsz,
        'data': str(data_root),
        'train': {}, 'val': {}, 'test': {}, 'exports': {}, 'artifacts': {},
    }

    best_pt_path: Path
    if skip_train:
        best_pt_path = Path(model_ref).resolve()
        if not best_pt_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {best_pt_path}')
        print(f'[skip-train] Using checkpoint: {best_pt_path}')
    else:
        print(f'[train] size={imgsz} data={data_root}')
        model = YOLO(model_ref)
        train_results = model.train(
            data=str(data_root), imgsz=imgsz, epochs=epochs, batch=batch, device=device,
            patience=patience, workers=workers, project=project, name=name, exist_ok=exist_ok,
        )
        summary['train'] = _safe_metrics(train_results)
        save_dir = Path(getattr(train_results, 'save_dir', Path(project) / name))
        best_pt_path = (save_dir / 'weights' / 'best.pt').resolve()
        if not best_pt_path.exists():
            raise FileNotFoundError(f'best.pt not found after training: {best_pt_path}')
        summary['artifacts']['best_pt'] = str(best_pt_path)

    best_model = YOLO(str(best_pt_path))
    val_results = best_model.val(data=str(data_root), imgsz=imgsz, split='val', device=device)
    test_results = best_model.val(data=str(data_root), imgsz=imgsz, split=test_split, device=device)
    summary['val'] = _safe_metrics(val_results)
    summary[test_split] = _safe_metrics(test_results)

    if not skip_export:
        onnx_out = best_model.export(format='onnx', imgsz=imgsz)
        summary['exports']['onnx'] = _stringify(onnx_out)
        if isinstance(onnx_out, (str, Path)):
            summary['artifacts']['onnx'] = str(onnx_out)
        ncnn_out = best_model.export(format='ncnn', imgsz=imgsz)
        summary['exports']['ncnn'] = _stringify(ncnn_out)
        if isinstance(ncnn_out, (str, Path)):
            summary['artifacts']['ncnn'] = str(ncnn_out)
    return summary


def parse_sizes(value: str) -> list[int]:
    out = []
    for tok in value.split(','):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    if not out:
        raise argparse.ArgumentTypeError('sizes must not be empty')
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description='Train Ultralytics classification model(s), validate, and export ONNX/NCNN.')
    parser.add_argument('--data', default='', help='Single dataset root')
    parser.add_argument('--data-template', default='', help='Dataset template with {size}, e.g. merged/px{size}')
    parser.add_argument('--model', default='yolo26n-cls.pt', help='Base model or checkpoint')
    parser.add_argument('--imgsz', type=int, default=100, help='Single input image size')
    parser.add_argument('--sizes', type=parse_sizes, default=None, help='Comma-separated sizes, e.g. 16,40,128')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--device', default='0')
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--project', default='runs/classify')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--test-split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--skip-export', action='store_true')
    parser.add_argument('--summary', default='train_export_summary.json')
    args = parser.parse_args()

    sizes = args.sizes or [args.imgsz]
    if args.data_template:
        dataset_for = lambda sz: Path(args.data_template.format(size=sz)).resolve()
    elif args.data:
        dataset_for = lambda sz: Path(args.data).resolve()
    else:
        raise SystemExit('Need --data or --data-template')

    summary: dict[str, Any] = {
        'args': vars(args),
        'runs': {},
        'recommended_deploy': None,
    }

    best_key = None
    best_top1 = -1.0
    for sz in sizes:
        data_root = dataset_for(sz)
        run_name = f'{args.name}_px{sz}' if len(sizes) > 1 else args.name
        run_summary = train_one_size(
            data_root=data_root,
            model_ref=args.model,
            imgsz=sz,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            patience=args.patience,
            workers=args.workers,
            project=args.project,
            name=run_name,
            exist_ok=args.exist_ok,
            test_split=args.test_split,
            skip_train=args.skip_train,
            skip_export=args.skip_export,
        )
        summary['runs'][f'px{sz}'] = _stringify(run_summary)
        try:
            top1 = float(run_summary.get('val', {}).get('top1', -1.0))
        except Exception:
            top1 = -1.0
        if top1 > best_top1:
            best_top1 = top1
            best_key = f'px{sz}'

    summary['recommended_deploy'] = best_key
    summary_path = Path(args.summary).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_stringify(summary), indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'recommended_deploy': best_key, 'summary': str(summary_path)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
