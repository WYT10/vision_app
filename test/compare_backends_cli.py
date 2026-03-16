#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Compare ONNX and NCNN using eval_classifier_cli.py outputs.')
    p.add_argument('--python', default='python3')
    p.add_argument('--eval-script', default='./eval_classifier_cli.py')
    p.add_argument('--exe', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--split', action='append', default=['test'])
    p.add_argument('--prep', default='crop')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--output-dir', default='bench_out_compare')
    p.add_argument('--onnx-model', required=True)
    p.add_argument('--ncnn-param', required=True)
    p.add_argument('--ncnn-bin', required=True)
    p.add_argument('--latency-image', default='')
    p.add_argument('--repeat', type=int, default=100)
    p.add_argument('--warmup', type=int, default=20)
    return p.parse_args()


def run(cmd):
    print('RUN:', ' '.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in args.split:
        for backend in ('onnx', 'ncnn'):
            cmd = [
                args.python, args.eval_script,
                '--exe', args.exe,
                '--backend', backend,
                '--labels', args.labels,
                '--dataset-root', args.dataset_root,
                '--split', split,
                '--prep', args.prep,
                '--threads', str(args.threads),
                '--output-dir', str(out),
                '--name', backend,
            ]
            if backend == 'onnx':
                cmd += ['--model', args.onnx_model]
            else:
                cmd += ['--model', args.ncnn_param, '--weights', args.ncnn_bin]
            run(cmd)

            summary_path = out / f'{backend}_{split}_summary.json'
            with summary_path.open('r', encoding='utf-8') as f:
                s = json.load(f)
            rows.append([
                backend, split, s['total_images'], f"{s['top1_acc']:.6f}", f"{s['top5_acc']:.6f}",
                f"{s['avg_wall_ms']:.3f}", f"{s['img_per_s']:.3f}"
            ])

    if args.latency_image:
        for backend in ('onnx', 'ncnn'):
            cmd = [
                args.python, args.eval_script,
                '--exe', args.exe,
                '--backend', backend,
                '--labels', args.labels,
                '--repeat-image', args.latency_image,
                '--repeat', str(args.repeat),
                '--warmup', str(args.warmup),
                '--prep', args.prep,
                '--threads', str(args.threads),
                '--output-dir', str(out),
                '--name', backend,
            ]
            if backend == 'onnx':
                cmd += ['--model', args.onnx_model]
            else:
                cmd += ['--model', args.ncnn_param, '--weights', args.ncnn_bin]
            run(cmd)
            summary_path = out / f'{backend}_repeat_repeat_summary.json'
            with summary_path.open('r', encoding='utf-8') as f:
                s = json.load(f)
            rows.append([
                backend, 'repeat_image', args.repeat, '', '', f"{s['mean_ms']:.3f}", f"{s['img_per_s']:.3f}"
            ])

    csv_path = out / 'compare_summary.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['backend', 'mode_or_split', 'count', 'top1_acc', 'top5_acc', 'avg_ms', 'img_per_s'])
        w.writerows(rows)

    md_path = out / 'compare_summary.md'
    with md_path.open('w', encoding='utf-8') as f:
        f.write('# Backend comparison\n\n')
        f.write('| backend | mode/split | count | top1_acc | top5_acc | avg_ms | img/s |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|\n')
        for r in rows:
            f.write(f'| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |\n')

    print(f'Wrote {csv_path}')
    print(f'Wrote {md_path}')
