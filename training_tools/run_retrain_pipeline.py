#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print('running:', ' '.join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)


def main() -> int:
    ap = argparse.ArgumentParser(description='Full synthetic + hard-example retrain pipeline for 16/40/128 models.')
    ap.add_argument('--src-dir', required=True, help='Synthetic source img_dataset root')
    ap.add_argument('--aug-config', default='', help='Optional aug_config.json from live_tune_aug.py; synthetic only')
    ap.add_argument('--run-dir', required=True, help='Controller session run dir with hard_examples/')
    ap.add_argument('--workspace-root', required=True, help='Root for synthetic / merged / runs / exports / reports')
    ap.add_argument('--base-model', default='yolo26n-cls.pt')
    ap.add_argument('--sizes', default='16,40,128')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--device', default='0')
    ap.add_argument('--include-low-conf', action='store_true')
    ap.add_argument('--max-hard-per-class', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    prep = project_root / 'training_tools' / 'prepare_cls_dataset.py'
    merge = project_root / 'training_tools' / 'merge_hard_examples.py'
    trainer = project_root / 'training_tools' / 'eval_export_cls.py'

    workspace = Path(args.workspace_root).resolve()
    (workspace / 'synthetic').mkdir(parents=True, exist_ok=True)
    (workspace / 'merged').mkdir(parents=True, exist_ok=True)
    (workspace / 'reports').mkdir(parents=True, exist_ok=True)

    size_list = [int(x.strip()) for x in args.sizes.split(',') if x.strip()]
    synthetic_dirs = {}
    merged_dirs = {}
    for sz in size_list:
        synthetic_name = f'px{sz}_synthetic'
        merged_name = f'px{sz}'
        run_cmd = [
            sys.executable, str(prep),
            '--src-dir', str(Path(args.src_dir).resolve()),
            '--output-root', str((workspace / 'synthetic').resolve()),
            '--target-size', str(sz),
            '--dataset-version', 'automation',
            '--name', synthetic_name,
            '--split-mode', 'by_source',
            '--seed', str(args.seed),
            '--allow-overwrite',
        ]
        if args.aug_config:
            run_cmd += ['--aug-config', str(Path(args.aug_config).resolve())]
        run(run_cmd)
        synthetic_dir = workspace / 'synthetic' / synthetic_name
        synthetic_dirs[sz] = synthetic_dir
        merged_dir = workspace / 'merged' / merged_name
        merge_cmd = [
            sys.executable, str(merge),
            '--base-dataset', str(synthetic_dir),
            '--run-dir', str(Path(args.run_dir).resolve()),
            '--out-dataset', str(merged_dir),
            '--max-per-class', str(args.max_hard_per_class),
            '--overwrite',
        ]
        if args.include_low_conf:
            merge_cmd.append('--include-low-conf')
        run(merge_cmd)
        merged_dirs[sz] = merged_dir

    train_cmd = [
        sys.executable, str(trainer),
        '--data-template', str((workspace / 'merged' / 'px{size}').resolve()),
        '--sizes', args.sizes,
        '--model', args.base_model,
        '--epochs', str(args.epochs),
        '--batch', str(args.batch),
        '--device', args.device,
        '--project', str((workspace / 'runs').resolve()),
        '--name', 'automation_refresh',
        '--exist-ok',
        '--summary', str((workspace / 'reports' / 'multi_size_summary.json').resolve()),
    ]
    run(train_cmd)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
