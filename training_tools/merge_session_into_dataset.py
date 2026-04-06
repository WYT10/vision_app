#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from PIL import Image, ImageOps

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

DEFAULT_LABEL_ALIASES: Dict[str, str] = {
    "A-枪支": "A_gun",
    "A_gun": "A_gun",
    "B-爆炸物": "B_explosive",
    "B_explosive": "B_explosive",
    "C-匕首": "C_dagger",
    "C_dagger": "C_dagger",
    "D-警棍": "D_baton",
    "D_baton": "D_baton",
    "E-消防斧": "E_fire_axe",
    "E_fire_axe": "E_fire_axe",
    "F-急救包": "F_first_aid_kit",
    "F_first_aid_kit": "F_first_aid_kit",
    "G-手电筒": "G_flashlight",
    "G_flashlight": "G_flashlight",
    "H-对讲机": "H_walkie_talkie",
    "H_walkie_talkie": "H_walkie_talkie",
    "I-防弹背心": "I_body_armor",
    "I_body_armor": "I_body_armor",
    "J-望远镜": "J_binoculars",
    "J_binoculars": "J_binoculars",
    "K-头盔": "K_helmet",
    "K_helmet": "K_helmet",
    "L-消防车": "L_fire_truck",
    "L_fire_truck": "L_fire_truck",
    "M-救护车": "M_ambulance",
    "M_ambulance": "M_ambulance",
    "N-装甲车": "N_armored_vehicle",
    "N_armored_vehicle": "N_armored_vehicle",
    "O-摩托车": "O_motorcycle",
    "O_motorcycle": "O_motorcycle",
}


@dataclass
class MergeStats:
    base_target_size: Optional[int] = None
    hard_examples_added: int = 0
    low_confidence_added: int = 0
    skipped_unknown_class: int = 0
    skipped_non_image: int = 0


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_label_aliases(base_dataset: Path, session_dir: Path) -> Dict[str, str]:
    aliases = dict(DEFAULT_LABEL_ALIASES)
    candidates = [
        session_dir / "label_aliases.json",
        session_dir.parent.parent / "label_aliases.json",  # output-root/label_aliases.json
        base_dataset / "label_aliases.json",
    ]
    for p in candidates:
        data = load_json_if_exists(p)
        if isinstance(data, dict):
            for k, v in data.items():
                aliases[str(k)] = str(v)
    return aliases


def canonical_label(label: str, aliases: Dict[str, str]) -> str:
    s = (label or "").strip()
    if s in aliases:
        return aliases[s]
    s2 = s.replace(" ", "_").replace("-", "_").strip("_")
    if s2 in aliases:
        return aliases[s2]
    m = re.match(r"^([A-Za-z][A-Za-z0-9]*)[-_\s].*$", s)
    if m:
        return m.group(1)
    return s2


def infer_target_size(base_dataset: Path, explicit: Optional[int]) -> Optional[int]:
    if explicit and explicit > 0:
        return explicit
    summary = load_json_if_exists(base_dataset / "dataset_summary.json")
    t = summary.get("target_size")
    return t if isinstance(t, int) and t > 0 else None


def read_labels(base_dataset: Path) -> list[str]:
    labels_path = base_dataset / "labels.txt"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.txt not found in base dataset: {labels_path}")
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise RuntimeError("labels.txt is empty")
    return labels


def copy_base_dataset(base_dataset: Path, out_dataset: Path, overwrite: bool) -> None:
    if out_dataset.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {out_dataset}. Use --overwrite to replace it.")
        shutil.rmtree(out_dataset)
    shutil.copytree(base_dataset, out_dataset)


def resize_and_save(src: Path, dst: Path, target_size: Optional[int]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if target_size is None:
        shutil.copy2(src, dst)
        return
    img = Image.open(src).convert("RGB")
    out = ImageOps.fit(
        img,
        (target_size, target_size),
        method=Image.Resampling.BILINEAR,
        centering=(0.5, 0.5),
    )
    out.save(dst, quality=95)


def merge_bucket(bucket_root: Path, out_train_root: Path, labels: list[str], aliases: Dict[str, str], target_size: Optional[int], prefix: str):
    added = 0
    skipped_unknown = 0
    skipped_non_image = 0
    label_set = set(labels)

    if not bucket_root.exists():
        return added, skipped_unknown, skipped_non_image

    for class_dir in sorted(bucket_root.iterdir()):
        if not class_dir.is_dir():
            continue
        dst_class = canonical_label(class_dir.name, aliases)
        if dst_class not in label_set:
            skipped_unknown += sum(1 for p in class_dir.rglob("*") if p.is_file())
            continue

        dst_dir = out_train_root / dst_class
        dst_dir.mkdir(parents=True, exist_ok=True)

        idx = 0
        for src in sorted(class_dir.rglob("*")):
            if not src.is_file():
                continue
            if not is_image_file(src):
                skipped_non_image += 1
                continue

            dst_name = f"{prefix}_{canonical_label(class_dir.name, aliases)}_{idx:06d}{src.suffix.lower()}"
            dst_path = dst_dir / dst_name
            n = 1
            while dst_path.exists():
                dst_name = f"{prefix}_{canonical_label(class_dir.name, aliases)}_{idx:06d}_{n}{src.suffix.lower()}"
                dst_path = dst_dir / dst_name
                n += 1

            resize_and_save(src, dst_path, target_size)
            added += 1
            idx += 1

    return added, skipped_unknown, skipped_non_image


def write_merge_summary(out_dataset: Path, stats: MergeStats, base_dataset: Path, session_dir: Path) -> None:
    summary = {
        "base_dataset": str(base_dataset),
        "session_dir": str(session_dir),
        "output_dataset": str(out_dataset),
        "base_target_size": stats.base_target_size,
        "hard_examples_added": stats.hard_examples_added,
        "low_confidence_added": stats.low_confidence_added,
        "skipped_unknown_class": stats.skipped_unknown_class,
        "skipped_non_image": stats.skipped_non_image,
    }
    (out_dataset / "merge_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Copy a generated dataset and merge session ROI images into train/ only."
    )
    p.add_argument("--base-dataset", required=True, help="Generated dataset root that already has train/ val/ test/ labels.txt")
    p.add_argument("--session-dir", required=True, help="Automation session folder that contains hard_examples/ and low_confidence/")
    p.add_argument("--out-dataset", required=True, help="Where to write the merged dataset")
    p.add_argument("--include-low-conf", type=int, default=1, help="1=merge low_confidence too, 0=hard_examples only")
    p.add_argument("--target-size", type=int, default=0, help="Resize merged ROI images to this square size. 0=infer from dataset_summary.json or copy as-is")
    p.add_argument("--overwrite", action="store_true", help="Delete out-dataset first if it already exists")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base_dataset = Path(args.base_dataset).resolve()
    session_dir = Path(args.session_dir).resolve()
    out_dataset = Path(args.out_dataset).resolve()

    if not base_dataset.exists():
        raise FileNotFoundError(f"Base dataset not found: {base_dataset}")
    if not session_dir.exists():
        raise FileNotFoundError(f"Session dir not found: {session_dir}")

    labels = read_labels(base_dataset)
    aliases = load_label_aliases(base_dataset, session_dir)
    target_size = infer_target_size(base_dataset, args.target_size if args.target_size > 0 else None)

    copy_base_dataset(base_dataset, out_dataset, overwrite=args.overwrite)

    out_train_root = out_dataset / "train"
    if not out_train_root.exists():
        raise FileNotFoundError(f"train/ not found after copying base dataset: {out_train_root}")

    stats = MergeStats(base_target_size=target_size)

    hard_added, hard_skip_unknown, hard_skip_non_img = merge_bucket(
        bucket_root=session_dir / "hard_examples",
        out_train_root=out_train_root,
        labels=labels,
        aliases=aliases,
        target_size=target_size,
        prefix="session_hard",
    )
    stats.hard_examples_added = hard_added
    stats.skipped_unknown_class += hard_skip_unknown
    stats.skipped_non_image += hard_skip_non_img

    if int(args.include_low_conf) == 1:
        low_added, low_skip_unknown, low_skip_non_img = merge_bucket(
            bucket_root=session_dir / "low_confidence",
            out_train_root=out_train_root,
            labels=labels,
            aliases=aliases,
            target_size=target_size,
            prefix="session_lowconf",
        )
        stats.low_confidence_added = low_added
        stats.skipped_unknown_class += low_skip_unknown
        stats.skipped_non_image += low_skip_non_img

    write_merge_summary(out_dataset, stats, base_dataset, session_dir)

    print("=" * 80)
    print("Merged dataset created")
    print(f"Base dataset : {base_dataset}")
    print(f"Session dir  : {session_dir}")
    print(f"Output       : {out_dataset}")
    print(f"Target size  : {target_size if target_size else 'copy as-is'}")
    print("-" * 80)
    print(f"hard_examples added   : {stats.hard_examples_added}")
    print(f"low_confidence added  : {stats.low_confidence_added}")
    print(f"skipped unknown class : {stats.skipped_unknown_class}")
    print(f"skipped non-image     : {stats.skipped_non_image}")
    print(f"summary json          : {out_dataset / 'merge_summary.json'}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
