#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

BEST_RE = re.compile(r"best:\s*\[(?P<idx>\d+)\]\s*(?P<label>[^\s]+)\s+prob=(?P<prob>[0-9.]+)")
TOPK_ITEM_RE = re.compile(r"\[(?P<idx>\d+)\](?P<label>[^=\s]+)=(?P<prob>[0-9.]+)")
PROCESSED_RE = re.compile(r"Processed\s+(?P<count>\d+)\s+image\(s\)\s+in\s+(?P<secs>[0-9.]+)\s+s\s+\((?P<imgps>[0-9.]+)\s+img/s\)")


@dataclass
class Prediction:
    image_path: str
    gt_label: str
    pred_label: str
    pred_idx: int
    pred_prob: float
    topk_labels: List[str]
    topk_probs: List[float]
    wall_time_s: float
    raw_stdout: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Evaluate portable_cls_infer over a classification folder tree.')
    p.add_argument('--exe', required=True, help='Path to portable_cls_infer executable')
    p.add_argument('--backend', required=True, choices=['onnx', 'ncnn'])
    p.add_argument('--model', required=True, help='Path to .onnx or .param model file')
    p.add_argument('--weights', default='', help='Path to .bin weights for NCNN')
    p.add_argument('--labels', required=True, help='Path to labels.txt')
    p.add_argument('--dataset-root', help='Root folder containing train/val/test subfolders')
    p.add_argument('--split', default='test', help='Dataset split under dataset-root (train/val/test)')
    p.add_argument('--input-root', help='Direct folder of class subfolders or a single image')
    p.add_argument('--prep', default='crop', choices=['crop', 'stretch', 'letterbox'])
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--topk', type=int, default=5)
    p.add_argument('--recursive', action='store_true', help='Recursively search images under input-root')
    p.add_argument('--repeat-image', help='Single image to benchmark repeatedly')
    p.add_argument('--repeat', type=int, default=1)
    p.add_argument('--warmup', type=int, default=0)
    p.add_argument('--output-dir', default='bench_out')
    p.add_argument('--name', default='', help='Optional run name prefix')
    p.add_argument('--print-every', type=int, default=25)
    return p.parse_args()


def collect_images(root: Path, recursive: bool) -> List[Path]:
    if root.is_file():
        return [root]
    iterator = root.rglob('*') if recursive else root.glob('*/*')
    files = []
    for p in iterator:
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return sorted(files)


def run_once(args: argparse.Namespace, image_path: Path) -> Prediction:
    cmd = [
        str(Path(args.exe)),
        '--backend', args.backend,
        '--model', args.model,
        '--labels', args.labels,
        '--input', str(image_path),
        '--prep', args.prep,
        '--threads', str(args.threads),
        '--topk', str(args.topk),
    ]
    if args.backend == 'ncnn':
        if not args.weights:
            raise ValueError('--weights is required for ncnn')
        cmd.extend(['--weights', args.weights])

    start = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    elapsed = time.perf_counter() - start
    out = proc.stdout

    best = BEST_RE.search(out)
    if not best:
        raise RuntimeError(f'Could not parse prediction for {image_path}\n--- output ---\n{out}')

    topk = TOPK_ITEM_RE.findall(out)
    topk_labels = [t[1] for t in topk]
    topk_probs = [float(t[2]) for t in topk]

    gt_label = image_path.parent.name if image_path.parent != image_path else ''

    return Prediction(
        image_path=str(image_path),
        gt_label=gt_label,
        pred_label=best.group('label'),
        pred_idx=int(best.group('idx')),
        pred_prob=float(best.group('prob')),
        topk_labels=topk_labels,
        topk_probs=topk_probs,
        wall_time_s=elapsed,
        raw_stdout=out,
    )


def benchmark_repeat(args: argparse.Namespace, image_path: Path) -> dict:
    for _ in range(args.warmup):
        _ = run_once(args, image_path)

    times = []
    last = None
    for _ in range(args.repeat):
        pred = run_once(args, image_path)
        last = pred
        times.append(pred.wall_time_s)

    mean_s = sum(times) / len(times)
    result = {
        'mode': 'repeat_image',
        'backend': args.backend,
        'image_path': str(image_path),
        'repeat': args.repeat,
        'warmup': args.warmup,
        'mean_s': mean_s,
        'mean_ms': mean_s * 1000.0,
        'img_per_s': 1.0 / mean_s if mean_s > 0 else 0.0,
        'min_ms': min(times) * 1000.0,
        'max_ms': max(times) * 1000.0,
        'pred_label': last.pred_label if last else '',
        'pred_prob': last.pred_prob if last else 0.0,
        'topk_labels': last.topk_labels if last else [],
        'topk_probs': last.topk_probs if last else [],
    }
    return result


def write_repeat_result(out_dir: Path, stem: str, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f'{stem}_repeat_summary.json').open('w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with (out_dir / f'{stem}_repeat_summary.md').open('w', encoding='utf-8') as f:
        f.write(f"# Repeat benchmark\n\n")
        for k, v in result.items():
            f.write(f"- **{k}**: {v}\n")


def write_eval_outputs(out_dir: Path, stem: str, preds: List[Prediction]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = sorted(set([p.gt_label for p in preds] + [p.pred_label for p in preds]))
    idx = {c: i for i, c in enumerate(classes)}
    confusion = [[0 for _ in classes] for _ in classes]

    total = len(preds)
    correct1 = 0
    correct5 = 0
    per_class_seen = Counter()
    per_class_correct1 = Counter()
    per_class_correct5 = Counter()

    for p in preds:
        per_class_seen[p.gt_label] += 1
        if p.pred_label == p.gt_label:
            correct1 += 1
            per_class_correct1[p.gt_label] += 1
        if p.gt_label in p.topk_labels[:5]:
            correct5 += 1
            per_class_correct5[p.gt_label] += 1
        confusion[idx[p.gt_label]][idx[p.pred_label]] += 1

    total_wall = sum(p.wall_time_s for p in preds)
    summary = {
        'mode': 'dataset_eval',
        'backend': args.backend,
        'total_images': total,
        'top1_correct': correct1,
        'top5_correct': correct5,
        'top1_acc': correct1 / total if total else 0.0,
        'top5_acc': correct5 / total if total else 0.0,
        'total_wall_s': total_wall,
        'avg_wall_ms': (total_wall / total * 1000.0) if total else 0.0,
        'img_per_s': (total / total_wall) if total_wall > 0 else 0.0,
        'classes': classes,
    }

    # predictions.csv
    with (out_dir / f'{stem}_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image_path', 'gt_label', 'pred_label', 'pred_idx', 'pred_prob', 'top5_labels', 'top5_probs', 'wall_time_ms'])
        for p in preds:
            w.writerow([
                p.image_path, p.gt_label, p.pred_label, p.pred_idx, f'{p.pred_prob:.6f}',
                '|'.join(p.topk_labels[:5]), '|'.join(f'{x:.6f}' for x in p.topk_probs[:5]), f'{p.wall_time_s*1000.0:.3f}'
            ])

    # per_class.csv
    with (out_dir / f'{stem}_per_class.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['class', 'seen', 'top1_correct', 'top1_acc', 'top5_correct', 'top5_acc'])
        for c in classes:
            seen = per_class_seen[c]
            c1 = per_class_correct1[c]
            c5 = per_class_correct5[c]
            w.writerow([c, seen, c1, f'{(c1/seen) if seen else 0.0:.6f}', c5, f'{(c5/seen) if seen else 0.0:.6f}'])

    # confusion_matrix.csv
    with (out_dir / f'{stem}_confusion_matrix.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['gt\\pred'] + classes)
        for i, c in enumerate(classes):
            w.writerow([c] + confusion[i])

    # summary json/md
    with (out_dir / f'{stem}_summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (out_dir / f'{stem}_summary.md').open('w', encoding='utf-8') as f:
        f.write(f"# Evaluation summary\n\n")
        f.write(f"- backend: **{summary['backend']}**\n")
        f.write(f"- total_images: **{summary['total_images']}**\n")
        f.write(f"- top1_acc: **{summary['top1_acc']:.4f}**\n")
        f.write(f"- top5_acc: **{summary['top5_acc']:.4f}**\n")
        f.write(f"- avg_wall_ms: **{summary['avg_wall_ms']:.3f}**\n")
        f.write(f"- img_per_s: **{summary['img_per_s']:.3f}**\n")

    print(f'Wrote: {out_dir / (stem + "_summary.json")}')
    print(f'Wrote: {out_dir / (stem + "_predictions.csv")}')
    print(f'Wrote: {out_dir / (stem + "_per_class.csv")}')
    print(f'Wrote: {out_dir / (stem + "_confusion_matrix.csv")}')


def make_stem(args: argparse.Namespace) -> str:
    base = args.name if args.name else args.backend
    if args.repeat_image:
        return f'{base}_repeat'
    split = args.split if args.dataset_root else Path(args.input_root).name if args.input_root else 'input'
    return f'{base}_{split}'


if __name__ == '__main__':
    args = parse_args()
    out_dir = Path(args.output_dir)
    stem = make_stem(args)

    if args.repeat_image:
        result = benchmark_repeat(args, Path(args.repeat_image))
        write_repeat_result(out_dir, stem, result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        sys.exit(0)

    if args.dataset_root:
        root = Path(args.dataset_root) / args.split
        recursive = True
    elif args.input_root:
        root = Path(args.input_root)
        recursive = args.recursive or root.is_file()
    else:
        raise SystemExit('Need --dataset-root or --input-root or --repeat-image')

    images = collect_images(root, recursive=recursive)
    if not images:
        raise SystemExit(f'No images found under {root}')

    preds: List[Prediction] = []
    for i, img in enumerate(images, start=1):
        pred = run_once(args, img)
        preds.append(pred)
        if i == 1 or i % max(1, args.print_every) == 0 or i == len(images):
            print(f'[{i}/{len(images)}] {img.name} gt={pred.gt_label} pred={pred.pred_label} p={pred.pred_prob:.4f} t={pred.wall_time_s*1000.0:.2f}ms')

    write_eval_outputs(out_dir, stem, preds)
