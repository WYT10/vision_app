#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def copy_tree(src: Path, dst: Path, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        if not overwrite:
            raise FileExistsError(dst)
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def merge_hard_examples(base_dataset: Path,
                        run_dir: Path,
                        out_dataset: Path,
                        include_low_conf: bool,
                        max_per_class: int,
                        overwrite: bool) -> dict:
    copy_tree(base_dataset, out_dataset, overwrite=overwrite)
    added = 0
    dedup = 0
    counts = {}
    seen_hashes = set()
    for bucket_name in ['hard_examples', 'low_confidence']:
        if bucket_name == 'low_confidence' and not include_low_conf:
            continue
        bucket = run_dir / bucket_name
        if not bucket.exists():
            continue
        for class_dir in sorted(bucket.iterdir()):
            if not class_dir.is_dir():
                continue
            train_dir = out_dataset / 'train' / class_dir.name
            train_dir.mkdir(parents=True, exist_ok=True)
            class_count = counts.get(class_dir.name, 0)
            for img in sorted(class_dir.iterdir()):
                if img.suffix.lower() not in IMG_EXTS:
                    continue
                digest = file_sha1(img)
                if digest in seen_hashes:
                    dedup += 1
                    continue
                if class_count >= max_per_class:
                    continue
                seen_hashes.add(digest)
                dst = train_dir / img.name
                if dst.exists():
                    dst = train_dir / f'{img.stem}__{digest[:8]}{img.suffix.lower()}'
                shutil.copy2(img, dst)
                class_count += 1
                added += 1
            counts[class_dir.name] = class_count
    meta = {'added': added, 'dedup_skipped': dedup, 'counts': counts}
    (out_dataset / 'merge_hard_examples_summary.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description='Merge collected hard examples into a base classification dataset.')
    ap.add_argument('--base-dataset', required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--out-dataset', required=True)
    ap.add_argument('--include-low-conf', action='store_true')
    ap.add_argument('--max-per-class', type=int, default=200)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    meta = merge_hard_examples(Path(args.base_dataset).resolve(), Path(args.run_dir).resolve(), Path(args.out_dataset).resolve(), args.include_low_conf, args.max_per_class, args.overwrite)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
