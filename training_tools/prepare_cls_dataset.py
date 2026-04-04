#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASS_NAME_MAP: Dict[str, str] = {
    "A-枪支": "A_gun",
    "B-爆炸物": "B_explosive",
    "C-匕首": "C_dagger",
    "D-警棍": "D_baton",
    "E-消防斧": "E_fire_axe",
    "F-急救包": "F_first_aid_kit",
    "G-手电筒": "G_flashlight",
    "H-对讲机": "H_walkie_talkie",
    "I-防弹背心": "I_body_armor",
    "J-望远镜": "J_binoculars",
    "K-头盔": "K_helmet",
    "L-消防车": "L_fire_truck",
    "M-救护车": "M_ambulance",
    "N-装甲车": "N_armored_vehicle",
    "O-摩托车": "O_motorcycle",
}

cv2.setUseOptimized(True)
cv2.setNumThreads(1)


def stable_seed(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def safe_name(name: str) -> str:
    s = name.strip()
    repl = {
        " ": "_", "-": "_", "/": "_", "\\": "_", "(": "", ")": "", "[": "", "]": "",
        "{": "", "}": "", ".": "_", ",": "_", ":": "_", ";": "_", "'": "", '"': "",
        "+": "plus", "#": "sharp",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def map_normal_class_name(src_class_name: str) -> str:
    return DEFAULT_CLASS_NAME_MAP.get(src_class_name, safe_name(src_class_name))


def list_class_dirs(src_dir: Path) -> List[Path]:
    return [p for p in sorted(src_dir.iterdir()) if p.is_dir() and not p.name.startswith('.')]


def list_images(folder: Path) -> List[Path]:
    return [p for p in sorted(folder.iterdir()) if p.is_file() and not p.name.startswith('.') and p.suffix.lower() in IMG_EXTS]


def imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def imwrite_unicode(path: Path, img: np.ndarray, jpg_quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    if not ok:
        raise RuntimeError(f"Failed to encode image for: {path}")
    buf.tofile(str(path))


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def pil_to_bgr(img_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def resize_to_target_pil(img_pil: Image.Image, target_size: int) -> Image.Image:
    return img_pil.resize((target_size, target_size), Image.Resampling.BILINEAR)


def affine_pil(img_pil: Image.Image, angle: float, translate: tuple[int, int], scale: float, shear_x: float) -> Image.Image:
    arr = np.array(img_pil)
    h, w = arr.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += translate[0]
    M[1, 2] += translate[1]
    if abs(shear_x) > 1e-9:
        sh = np.tan(np.deg2rad(shear_x))
        S = np.array([[1.0, sh, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        M3 = np.vstack([M, [0.0, 0.0, 1.0]])
        S3 = np.vstack([S, [0.0, 0.0, 1.0]])
        M = (S3 @ M3)[:2, :]
    warped = cv2.warpAffine(arr, M.astype(np.float32), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return Image.fromarray(warped)


def apply_salt_pepper_noise(img_pil: Image.Image, rng: random.Random, amount_range: Sequence[float], salt_vs_pepper: float, dot_size: int = 1) -> Image.Image:
    img = np.array(img_pil).copy()
    h, w, _ = img.shape
    amount = rng.uniform(*amount_range)
    total_pixels = h * w
    noisy_pixels = max(1, int(total_pixels * amount))
    num_dots = max(1, int(noisy_pixels / max(1, dot_size * dot_size)))
    num_salt = int(num_dots * salt_vs_pepper)
    num_pepper = max(0, num_dots - num_salt)
    for _ in range(num_salt):
        y = rng.randrange(h)
        x = rng.randrange(w)
        img[max(0, y):min(h, y + dot_size), max(0, x):min(w, x + dot_size)] = 255
    for _ in range(num_pepper):
        y = rng.randrange(h)
        x = rng.randrange(w)
        img[max(0, y):min(h, y + dot_size), max(0, x):min(w, x + dot_size)] = 0
    return Image.fromarray(img)


def apply_median_blur(img_pil: Image.Image, rng: random.Random, kernel_choices: Sequence[int]) -> Image.Image:
    k = int(rng.choice(list(kernel_choices)))
    if k % 2 == 0:
        k += 1
    k = max(1, k)
    arr = np.array(img_pil)
    arr = cv2.medianBlur(arr, k)
    return Image.fromarray(arr)


def apply_post_aug_ops(img_pil: Image.Image, rng: random.Random, op_sequence: Sequence[str], amount_range: Sequence[float], salt_vs_pepper: float, kernel_choices: Sequence[int]) -> Image.Image:
    img = img_pil
    for op in op_sequence:
        if op == 'salt_pepper_noise':
            img = apply_salt_pepper_noise(img, rng, amount_range, salt_vs_pepper)
        elif op == 'median_blur':
            img = apply_median_blur(img, rng, kernel_choices)
        elif op in ('', 'none'):
            continue
        else:
            raise ValueError(f'Unknown post augmentation op: {op}')
    return img


def apply_tuned_sequence(img_pil: Image.Image, rng: random.Random, sequence: Sequence[dict], final_size: int) -> Image.Image:
    img = img_pil
    for op in sequence:
        t = op['type']
        if t == 'sp':
            img = apply_salt_pepper_noise(
                img,
                rng=random.Random(int(op.get('seed', 42))),
                amount_range=(float(op['amount']), float(op['amount'])),
                salt_vs_pepper=float(op['ratio']),
                dot_size=int(op.get('dot_size', 2)),
            )
        elif t == 'mb':
            img = apply_median_blur(img, rng, [int(op['k'])])
        elif t == 'rs':
            img = resize_to_target_pil(img, int(op['size']))
        elif t == 'af':
            w, h = img.size
            translate = [int(float(op['tx']) * w), int(float(op['ty']) * h)]
            img = affine_pil(img, float(op['angle']), tuple(translate), float(op['scale']), float(op['shear']))
        else:
            raise ValueError(f'Unknown tuned op type: {t}')
    return resize_to_target_pil(img, final_size)


def augment_default(base_pil: Image.Image, seed: int, target_size: int, post_ops: Sequence[str], sp_amount_range: Sequence[float], salt_vs_pepper: float, kernel_choices: Sequence[int]) -> Image.Image:
    rng = random.Random(seed)
    img = resize_to_target_pil(base_pil, target_size)
    angle = rng.uniform(-20.0, 20.0)
    translate = (int(rng.uniform(-0.06, 0.06) * target_size), int(rng.uniform(-0.06, 0.06) * target_size))
    scale = rng.uniform(0.92, 1.08)
    shear = [rng.uniform(-6.0, 6.0), 0.0]
    img = affine_pil(img, angle=angle, translate=translate, scale=scale, shear_x=shear[0])
    if rng.random() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.88, 1.12))
    if rng.random() < 0.8:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.88, 1.12))
    if rng.random() < 0.5:
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.9, 1.1))
    img = apply_post_aug_ops(img, rng, post_ops, sp_amount_range, salt_vs_pepper, kernel_choices)
    return resize_to_target_pil(img, target_size)


def load_aug_config(path: Path | None) -> tuple[bool, list[dict]]:
    if path is None or not path.exists():
        return False, []
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('custom_sequence'):
        return True, list(data.get('sequence', []))
    return False, []


def split_sources(images: List[Path], mode: str, seed: int) -> Dict[str, List[Path]]:
    if mode == 'leaked':
        return {'train': images, 'val': images, 'test': images}
    rng = random.Random(seed)
    pool = images[:]
    rng.shuffle(pool)
    n = len(pool)
    n_train = max(1, int(round(n * 0.7))) if n >= 3 else max(1, n - 2)
    n_val = max(1, int(round(n * 0.15))) if n >= 6 else 1 if n >= 2 else 0
    if n_train + n_val >= n:
        n_val = max(0, min(n_val, n - n_train - 1))
    train = pool[:n_train]
    val = pool[n_train:n_train + n_val]
    test = pool[n_train + n_val:]
    if not test and val:
        test = [val.pop()]
    if not val and len(train) > 1:
        val = [train.pop()]
    return {'train': train, 'val': val, 'test': test}


def build_versions_for_split(src_path: Path, split_name: str, out_count: int, target_size: int, seed: int, tuned_custom: bool, tuned_sequence: Sequence[dict], post_ops: Sequence[str], sp_amount_range: Sequence[float], salt_vs_pepper: float, kernel_choices: Sequence[int]) -> List[np.ndarray]:
    src_bgr = imread_unicode(src_path)
    base_pil = bgr_to_pil(src_bgr)
    versions: List[np.ndarray] = []
    split_seed_offset = {'train': 100000, 'val': 200000, 'test': 300000}[split_name]
    base_seed = seed + stable_seed(str(src_path)) + split_seed_offset
    for i in range(out_count):
        if i == 0:
            img_pil = resize_to_target_pil(base_pil, target_size)
        else:
            if tuned_custom:
                img_pil = apply_tuned_sequence(base_pil, random.Random(base_seed + i), tuned_sequence, target_size)
            else:
                img_pil = augment_default(base_pil, base_seed + i, target_size, post_ops, sp_amount_range, salt_vs_pepper, kernel_choices)
        versions.append(pil_to_bgr(img_pil))
    return versions


def write_metadata(out_dir: Path, summary: dict) -> None:
    class_names = sorted(summary['classes'].keys())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    (out_dir / 'labels.txt').write_text('\n'.join(class_names), encoding='utf-8')
    (out_dir / 'class_to_idx.json').write_text(json.dumps(class_to_idx, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'dataset_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> int:
    ap = argparse.ArgumentParser(description='Prepare a synthetic classification dataset for one target size.')
    ap.add_argument('--src-dir', required=True, help='Source root with class subfolders')
    ap.add_argument('--output-root', required=True, help='Folder that will contain generated dataset folders')
    ap.add_argument('--target-size', type=int, required=True, help='Final square image size, e.g. 16/40/128')
    ap.add_argument('--dataset-version', default='v_cli')
    ap.add_argument('--dataset-prefix', default='dataset_cls')
    ap.add_argument('--versions-train', type=int, default=5)
    ap.add_argument('--versions-val', type=int, default=5)
    ap.add_argument('--versions-test', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--split-mode', choices=['leaked', 'by_source'], default='by_source')
    ap.add_argument('--aug-config', default='', help='Optional aug_config.json from live_tune_aug.py; synthetic only')
    ap.add_argument('--post-ops', default='salt_pepper_noise,median_blur')
    ap.add_argument('--sp-amount-range', default='0.003,0.02')
    ap.add_argument('--sp-salt-vs-pepper', type=float, default=0.5)
    ap.add_argument('--median-kernels', default='3,5,7')
    ap.add_argument('--jpg-quality', type=int, default=92)
    ap.add_argument('--max-workers', type=int, default=max(1, min(8, (os.cpu_count() or 8) - 2)))
    ap.add_argument('--allow-overwrite', action='store_true')
    ap.add_argument('--name', default='')
    args = ap.parse_args()

    src_dir = Path(args.src_dir).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tuned_custom, tuned_sequence = load_aug_config(Path(args.aug_config).resolve() if args.aug_config else None)
    run_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    versions = {'train': args.versions_train, 'val': args.versions_val, 'test': args.versions_test}
    folder_name = args.name or (
        f"{args.dataset_prefix}__{args.dataset_version}__dt{run_tag}__px{args.target_size}__"
        f"tr{versions['train']}__va{versions['val']}__te{versions['test']}__seed{args.seed}__{args.split_mode}"
    )
    out_dir = output_root / folder_name
    if out_dir.exists():
        if not args.allow_overwrite:
            raise FileExistsError(f'Output directory already exists: {out_dir}')
        shutil.rmtree(out_dir)
    for split in ('train', 'val', 'test'):
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    sp_lo, sp_hi = [float(x) for x in args.sp_amount_range.split(',')]
    post_ops = [x.strip() for x in args.post_ops.split(',') if x.strip()]
    kernel_choices = [max(1, int(x)) for x in args.median_kernels.split(',') if x.strip()]

    class_dirs = list_class_dirs(src_dir)
    if not class_dirs:
        raise RuntimeError(f'No class folders found in {src_dir}')

    summary = {
        'source_dir': str(src_dir),
        'output_dir': str(out_dir),
        'target_size': args.target_size,
        'versions_per_split': versions,
        'seed': args.seed,
        'split_mode': args.split_mode,
        'is_custom_tuned_sequence': tuned_custom,
        'tuned_sequence': tuned_sequence if tuned_custom else [],
        'post_aug_op_sequence': post_ops,
        'classes': {},
        'totals': {'train': 0, 'val': 0, 'test': 0, 'all': 0},
        'warnings': [],
    }

    print('=' * 84)
    print('Preparing classification dataset')
    print(f'Source: {src_dir}')
    print(f'Output dir: {out_dir}')
    print(f'Target size: {args.target_size}x{args.target_size}')
    print(f'Split mode: {args.split_mode}')
    print(f'Versions per split: {versions}')
    if tuned_custom:
        print(f'[!] Using custom sequence from {args.aug_config} with {len(tuned_sequence)} ops')
    else:
        print(f'Post-aug op sequence: {post_ops}')
    print('=' * 84)

    for class_dir in tqdm(class_dirs, desc='Classes', leave=True):
        out_class_name = map_normal_class_name(class_dir.name)
        images = list_images(class_dir)
        split_map = split_sources(images, args.split_mode, args.seed + stable_seed(class_dir.name))
        class_counts = {'train': 0, 'val': 0, 'test': 0}
        for split_name, split_images in split_map.items():
            if not split_images:
                continue
            out_class_dir = out_dir / split_name / out_class_name
            out_class_dir.mkdir(parents=True, exist_ok=True)
            with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
                futures = {}
                for i, src_path in enumerate(split_images):
                    futures[ex.submit(
                        build_versions_for_split,
                        src_path, split_name, versions[split_name], args.target_size, args.seed,
                        tuned_custom, tuned_sequence, post_ops, (sp_lo, sp_hi), args.sp_salt_vs_pepper, kernel_choices,
                    )] = (i, src_path)
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f'{out_class_name}:{split_name}', leave=False):
                    i, src_path = futures[fut]
                    imgs = fut.result()
                    for j, img in enumerate(imgs):
                        dst_path = out_class_dir / f"{out_class_name}_{i:05d}_{split_name}_{j}.jpg"
                        imwrite_unicode(dst_path, img, jpg_quality=args.jpg_quality)
                        class_counts[split_name] += 1
        total = sum(class_counts.values())
        summary['classes'][out_class_name] = {
            'source_name': class_dir.name,
            'source_count_total': len(images),
            'source_split_counts': {k: len(v) for k, v in split_map.items()},
            'total': total,
            **class_counts,
        }
        for split_name in ('train', 'val', 'test'):
            summary['totals'][split_name] += class_counts[split_name]
        summary['totals']['all'] += total

    if args.split_mode == 'leaked':
        summary['warnings'].append('split_mode=leaked copies the same source into train/val/test and can inflate metrics.')

    write_metadata(out_dir, summary)
    print('=' * 84)
    print('Classification dataset prepared')
    print(f'Output dir: {out_dir}')
    print(f'Total images: {summary["totals"]["all"]}')
    print('=' * 84)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
