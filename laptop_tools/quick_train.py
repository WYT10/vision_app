#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description='Convenience wrapper around the multi-size retrain pipeline.')
    ap.add_argument('--src-dir', required=True, help='Synthetic source img_dataset root')
    ap.add_argument('--run-dir', required=True, help='Controller session run dir')
    ap.add_argument('--workspace-root', default='', help='Optional. Default: sibling workspaces/<session> next to run-dir')
    ap.add_argument('--base-model', default='yolo26n-cls.pt')
    ap.add_argument('--aug-config', default='')
    ap.add_argument('--sizes', default='16,40,128')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--device', default='0')
    ap.add_argument('--include-low-conf', action='store_true')
    ap.add_argument('--max-hard-per-class', type=int, default=200)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    pipeline = project_root / 'training_tools' / 'run_retrain_pipeline.py'
    run_dir = Path(args.run_dir).resolve()
    workspace_root = Path(args.workspace_root).resolve() if args.workspace_root else (run_dir.parents[1] / 'workspaces' / run_dir.name)
    cmd = [
        sys.executable, str(pipeline),
        '--src-dir', str(Path(args.src_dir).resolve()),
        '--run-dir', str(run_dir),
        '--workspace-root', str(workspace_root),
        '--base-model', args.base_model,
        '--sizes', args.sizes,
        '--epochs', str(args.epochs),
        '--batch', str(args.batch),
        '--device', args.device,
        '--max-hard-per-class', str(args.max_hard_per_class),
    ]
    if args.aug_config:
        cmd += ['--aug-config', str(Path(args.aug_config).resolve())]
    if args.include_low_conf:
        cmd.append('--include-low-conf')
    print('running:', ' '.join(cmd))
    return subprocess.call(cmd)


if __name__ == '__main__':
    raise SystemExit(main())
