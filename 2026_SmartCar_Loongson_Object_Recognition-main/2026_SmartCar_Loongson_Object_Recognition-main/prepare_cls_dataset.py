from __future__ import annotations

import json
import os
import random
import shutil
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

# =========================================================
# Config
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "img_dataset"

# Output folder management
OUTPUT_ROOT = PROJECT_ROOT / "generated_datasets"
OUTPUT_PREFIX = "dataset_cls"
DATASET_VERSION = "v3"
AUTO_NAME_OUTPUT = True

SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPECIAL_CHAR_FOLDER = "字母和数字标识"

# Final ROI / model input size
TARGET_SIZE = 60

# REQUIRED BEHAVIOR:
# each source image creates 5 images in each split
VERSIONS_PER_SPLIT = {
    "train": 5,
    "val": 5,
    "test": 5,
}

MAX_WORKERS = max(1, min(8, (os.cpu_count() or 8) - 2))

CLASS_NAME_MAP: Dict[str, str] = {
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


# =========================================================
# Folder naming
# =========================================================
def build_dataset_folder_name() -> str:
    return (
        f"{OUTPUT_PREFIX}"
        f"__{DATASET_VERSION}"
        f"__px{TARGET_SIZE}"
        f"__tr{VERSIONS_PER_SPLIT['train']}"
        f"__va{VERSIONS_PER_SPLIT['val']}"
        f"__te{VERSIONS_PER_SPLIT['test']}"
        f"__seed{SEED}"
    )


if AUTO_NAME_OUTPUT:
    OUT_DIR = OUTPUT_ROOT / build_dataset_folder_name()
else:
    OUT_DIR = PROJECT_ROOT / "dataset_cls"


# =========================================================
# Utilities
# =========================================================
def stable_seed(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def list_class_dirs(src_dir: Path) -> List[Path]:
    return [p for p in sorted(src_dir.iterdir()) if p.is_dir() and not p.name.startswith(".")]


def list_images(folder: Path) -> List[Path]:
    return [
        p for p in sorted(folder.iterdir())
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMG_EXTS
    ]


def safe_name(name: str) -> str:
    s = name.strip()
    repl = {
        " ": "_",
        "-": "_",
        "/": "_",
        "\\": "_",
        "(": "",
        ")": "",
        "[": "",
        "]": "",
        "{": "",
        "}": "",
        ".": "_",
        ",": "_",
        ":": "_",
        ";": "_",
        "'": "",
        '"': "",
        "+": "plus",
        "#": "sharp",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def map_normal_class_name(src_class_name: str) -> str:
    return CLASS_NAME_MAP.get(src_class_name, safe_name(src_class_name))


def map_char_class_name(img_path: Path) -> str:
    return f"char_{safe_name(img_path.stem)}"


def ensure_clean_output(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (out_dir / split).mkdir(parents=True, exist_ok=True)


def imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def imwrite_unicode(path: Path, img: np.ndarray, jpg_quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    if not ok:
        raise RuntimeError(f"Failed to encode image for: {path}")
    buf.tofile(str(path))


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def pil_to_bgr(img_pil: Image.Image) -> np.ndarray:
    img_rgb = np.array(img_pil)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def resize_to_target_pil(img_pil: Image.Image) -> Image.Image:
    return TF.resize(
        img_pil,
        [TARGET_SIZE, TARGET_SIZE],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )


# =========================================================
# Torchvision-style augmentation
# =========================================================
def augment_with_torchvision(base_pil: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)

    img = resize_to_target_pil(base_pil)

    angle = rng.uniform(-15.0, 15.0)
    translate = (
        int(rng.uniform(-0.06, 0.06) * TARGET_SIZE),
        int(rng.uniform(-0.06, 0.06) * TARGET_SIZE),
    )
    scale = rng.uniform(0.92, 1.08)
    shear = [rng.uniform(-6.0, 6.0), 0.0]

    img = TF.affine(
        img,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=shear,
        interpolation=InterpolationMode.BILINEAR,
        fill=0,
    )

    if rng.random() < 0.8:
        img = TF.adjust_brightness(img, rng.uniform(0.88, 1.12))
    if rng.random() < 0.8:
        img = TF.adjust_contrast(img, rng.uniform(0.88, 1.12))
    if rng.random() < 0.5:
        img = TF.adjust_saturation(img, rng.uniform(0.9, 1.1))

    if rng.random() < 0.5:
        k = rng.choice([3, 5, 7])
        sigma = rng.uniform(0.2, 1.5)
        img = TF.gaussian_blur(img, kernel_size=[k, k], sigma=[sigma, sigma])

    return img


def build_versions_for_split(src_path: Path, split_name: str, out_count: int) -> List[np.ndarray]:
    """
    For one source image and one split:
      - version 0 = original resized
      - versions 1..N-1 = transformed
    All outputs are TARGET_SIZE x TARGET_SIZE.
    """
    src_bgr = imread_unicode(src_path)
    base_pil = bgr_to_pil(src_bgr)

    versions: List[np.ndarray] = []

    # split-specific seed offsets
    split_seed_offset = {
        "train": 100000,
        "val": 200000,
        "test": 300000,
    }[split_name]

    base_seed = SEED + stable_seed(str(src_path)) + split_seed_offset

    for i in range(out_count):
        if i == 0:
            img_pil = resize_to_target_pil(base_pil)
        else:
            img_pil = augment_with_torchvision(base_pil, base_seed + i)
            img_pil = resize_to_target_pil(img_pil)
        versions.append(pil_to_bgr(img_pil))

    return versions


# =========================================================
# Dataset builders
# =========================================================
def process_source_to_all_splits(src_path: Path, out_class_name: str, source_index: int) -> Dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}

    for split_name in ("train", "val", "test"):
        out_dir = OUT_DIR / split_name / out_class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        versions = build_versions_for_split(
            src_path,
            split_name=split_name,
            out_count=VERSIONS_PER_SPLIT[split_name],
        )

        for j, img in enumerate(versions):
            dst_path = out_dir / f"{out_class_name}_{source_index:05d}_{split_name}_{j}.jpg"
            imwrite_unicode(dst_path, img)
            counts[split_name] += 1

    return counts


def process_normal_class(class_dir: Path, out_class_name: str) -> dict:
    """
    Every source image appears in train/val/test.
    Each split gets 5 versions of that source image.
    WARNING: this causes split leakage.
    """
    images = list_images(class_dir)
    out_counts = {"train": 0, "val": 0, "test": 0}

    with tqdm(total=len(images), desc=f"{out_class_name}", leave=False) as pbar:
        for i, src_path in enumerate(images):
            counts = process_source_to_all_splits(src_path, out_class_name, i)
            out_counts["train"] += counts["train"]
            out_counts["val"] += counts["val"]
            out_counts["test"] += counts["test"]
            pbar.update(1)

    total_versions_per_source = (
        VERSIONS_PER_SPLIT["train"] +
        VERSIONS_PER_SPLIT["val"] +
        VERSIONS_PER_SPLIT["test"]
    )

    return {
        "class_name": out_class_name,
        "source_name": class_dir.name,
        "total": len(images) * total_versions_per_source,
        "train": out_counts["train"],
        "val": out_counts["val"],
        "test": out_counts["test"],
        "warning": (
            f"{out_class_name}: each source image appears in train/val/test; "
            f"this causes split leakage and optimistic metrics."
        ),
    }


def process_one_char_image(src_path: Path) -> dict:
    """
    Special folder behavior remains:
    each image becomes its own class, and now that class appears in train/val/test,
    with 5 versions in each split.
    """
    out_class_name = map_char_class_name(src_path)
    counts = process_source_to_all_splits(src_path, out_class_name, 0)

    total_versions_per_source = (
        VERSIONS_PER_SPLIT["train"] +
        VERSIONS_PER_SPLIT["val"] +
        VERSIONS_PER_SPLIT["test"]
    )

    return {
        "class_name": out_class_name,
        "source_name": f"{SPECIAL_CHAR_FOLDER}/{src_path.name}",
        "total": total_versions_per_source,
        "train": counts["train"],
        "val": counts["val"],
        "test": counts["test"],
        "warning": (
            f"{out_class_name}: one source image is expanded into train/val/test; "
            f"metrics measure transform robustness, not true generalization."
        ),
    }


def write_metadata(summary: dict) -> None:
    class_names = sorted(summary["classes"].keys())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    (OUT_DIR / "labels.txt").write_text("\n".join(class_names), encoding="utf-8")
    (OUT_DIR / "class_to_idx.json").write_text(
        json.dumps(class_to_idx, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SRC_DIR}")

    class_dirs = list_class_dirs(SRC_DIR)
    if not class_dirs:
        raise RuntimeError(f"No class folders found in: {SRC_DIR}")

    ensure_clean_output(OUT_DIR)

    summary = {
        "source_dir": str(SRC_DIR),
        "output_root": str(OUTPUT_ROOT),
        "output_dir": str(OUT_DIR),
        "dataset_folder_name": OUT_DIR.name,
        "dataset_version": DATASET_VERSION,
        "seed": SEED,
        "target_size": TARGET_SIZE,
        "versions_per_split": VERSIONS_PER_SPLIT,
        "classes": {},
        "totals": {"train": 0, "val": 0, "test": 0, "all": 0},
        "warnings": [],
    }

    normal_class_dirs = [d for d in class_dirs if d.name != SPECIAL_CHAR_FOLDER]

    print("=" * 84)
    print("Preparing classification dataset")
    print(f"Source: {SRC_DIR}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Folder name: {OUT_DIR.name}")
    print(f"Target size: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"Versions per split: {VERSIONS_PER_SPLIT}")
    print("=" * 84)

    # Normal classes
    for class_dir in tqdm(normal_class_dirs, desc="Normal classes", leave=True):
        out_class_name = map_normal_class_name(class_dir.name)
        result = process_normal_class(class_dir, out_class_name)

        summary["classes"][result["class_name"]] = {
            "source_name": result["source_name"],
            "total": result["total"],
            "train": result["train"],
            "val": result["val"],
            "test": result["test"],
        }
        summary["totals"]["train"] += result["train"]
        summary["totals"]["val"] += result["val"]
        summary["totals"]["test"] += result["test"]
        summary["totals"]["all"] += result["total"]
        if result["warning"]:
            summary["warnings"].append(result["warning"])

    # Character classes
    char_dir = SRC_DIR / SPECIAL_CHAR_FOLDER
    if char_dir.exists() and char_dir.is_dir():
        char_images = list_images(char_dir)

        if char_images:
            print("-" * 84)
            print(f"Processing special folder: {SPECIAL_CHAR_FOLDER}")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(process_one_char_image, p) for p in char_images]

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Char classes", leave=True):
                result = fut.result()
                summary["classes"][result["class_name"]] = {
                    "source_name": result["source_name"],
                    "total": result["total"],
                    "train": result["train"],
                    "val": result["val"],
                    "test": result["test"],
                }
                summary["totals"]["train"] += result["train"]
                summary["totals"]["val"] += result["val"]
                summary["totals"]["test"] += result["test"]
                summary["totals"]["all"] += result["total"]
                if result["warning"]:
                    summary["warnings"].append(result["warning"])

    write_metadata(summary)

    print("=" * 84)
    print("Classification dataset prepared")
    print(f"Source: {SRC_DIR}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Folder name: {OUT_DIR.name}")
    print(f"Target size: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"Versions per split: {VERSIONS_PER_SPLIT}")
    print("=" * 84)

    for cls_name in sorted(summary["classes"].keys()):
        s = summary["classes"][cls_name]
        print(
            f"{cls_name:24s} | total={s['total']:4d} "
            f"train={s['train']:4d} val={s['val']:4d} test={s['test']:4d}"
        )

    print("-" * 84)
    print(
        f"TOTAL | all={summary['totals']['all']} "
        f"train={summary['totals']['train']} "
        f"val={summary['totals']['val']} "
        f"test={summary['totals']['test']}"
    )

    if summary["warnings"]:
        print("-" * 84)
        print("WARNINGS:")
        for w in summary["warnings"]:
            print(f"  - {w}")

    print("-" * 84)
    print("Done.")


if __name__ == "__main__":
    main()  