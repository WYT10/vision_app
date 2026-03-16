from __future__ import annotations

import json
import math
import os
import random
import shutil
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# =========================================================
# Speed-oriented config
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "img_dataset"
OUT_DIR = PROJECT_ROOT / "dataset_cls"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SEED = 42
COPY_FILES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPECIAL_CHAR_FOLDER = "字母和数字标识"

# Keep this modest. Bigger = slower and usually not much better.
CHAR_TARGET_COUNT = 24

# Large blur, but not crazy
BLUR_KERNEL_CHOICES = [7, 9, 11]

# Parallelism for character generation
MAX_WORKERS = max(1, min(8, (os.cpu_count() or 8) - 2))

# Normal class folder -> output class name
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

# Avoid OpenCV internally spawning too many threads on top of ThreadPoolExecutor
cv2.setUseOptimized(True)
cv2.setNumThreads(1)


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


def compute_split_counts(n: int) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0

    n_train = int(round(n * TRAIN_RATIO))
    n_val = int(round(n * VAL_RATIO))
    n_test = n - n_train - n_val

    while n_train + n_val + n_test < n:
        n_train += 1
    while n_train + n_val + n_test > n:
        if n_train > 1:
            n_train -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            n_test -= 1

    if n >= 3:
        if n_val == 0 and n_train > 1:
            n_train -= 1
            n_val += 1
        if n_test == 0 and n_train > 1:
            n_train -= 1
            n_test += 1

    return n_train, n_val, n_test


def ensure_clean_output(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val", "test"):
        (out_dir / split).mkdir(parents=True, exist_ok=True)


def copy_or_move(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if COPY_FILES:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def imwrite_unicode(path: Path, img: np.ndarray, jpg_quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(
        ".jpg",
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality],
    )
    if not ok:
        raise RuntimeError(f"Failed to encode image for: {path}")
    buf.tofile(str(path))


def estimate_bg_color(img: np.ndarray) -> Tuple[int, int, int]:
    h, w = img.shape[:2]
    patch = max(2, min(h, w) // 12)
    corners = np.concatenate([
        img[:patch, :patch].reshape(-1, 3),
        img[:patch, -patch:].reshape(-1, 3),
        img[-patch:, :patch].reshape(-1, 3),
        img[-patch:, -patch:].reshape(-1, 3),
    ], axis=0)
    med = np.median(corners, axis=0)
    return tuple(int(x) for x in med.tolist())


def add_margin(img: np.ndarray, frac: float = 0.08) -> np.ndarray:
    h, w = img.shape[:2]
    pad_y = int(h * frac)
    pad_x = int(w * frac)
    bg = estimate_bg_color(img)
    return cv2.copyMakeBorder(
        img, pad_y, pad_y, pad_x, pad_x,
        borderType=cv2.BORDER_CONSTANT,
        value=bg
    )


# =========================================================
# Faster char augmentation
# =========================================================
def random_affine_fast(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    angle = rng.uniform(-16, 16)
    scale = rng.uniform(0.88, 1.12)
    tx = rng.uniform(-0.08, 0.08) * w
    ty = rng.uniform(-0.08, 0.08) * h

    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    bg = estimate_bg_color(img)
    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg,
    )


def random_perspective_light(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    dx = 0.04 * w
    dy = 0.04 * h

    src = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ])

    dst = np.float32([
        [rng.uniform(0, dx), rng.uniform(0, dy)],
        [w - 1 - rng.uniform(0, dx), rng.uniform(0, dy)],
        [w - 1 - rng.uniform(0, dx), h - 1 - rng.uniform(0, dy)],
        [rng.uniform(0, dx), h - 1 - rng.uniform(0, dy)],
    ])

    H = cv2.getPerspectiveTransform(src, dst)
    bg = estimate_bg_color(img)
    return cv2.warpPerspective(
        img,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg,
    )


def brightness_contrast(img: np.ndarray, rng: random.Random) -> np.ndarray:
    alpha = rng.uniform(0.85, 1.18)
    beta = rng.uniform(-20, 20)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def add_noise(img: np.ndarray, np_rng: np.random.Generator) -> np.ndarray:
    sigma = np_rng.uniform(1.5, 7.0)
    noise = np_rng.normal(0, sigma, img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def resize_blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    scale = rng.uniform(0.55, 0.85)
    sw = max(8, int(w * scale))
    sh = max(8, int(h * scale))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def motion_blur(img: np.ndarray, k: int, angle_deg: float) -> np.ndarray:
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0

    center = (k / 2 - 0.5, k / 2 - 0.5)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))

    s = kernel.sum()
    if s <= 0:
        kernel[k // 2, :] = 1.0
        s = kernel.sum()

    kernel /= s
    return cv2.filter2D(img, -1, kernel)


def strong_blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    k = rng.choice(BLUR_KERNEL_CHOICES)
    mode = rng.choice(["gaussian", "box", "motion"])

    if mode == "gaussian":
        sigma = rng.uniform(0.8, 2.0)
        return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)
    if mode == "box":
        return cv2.blur(img, (k, k))
    return motion_blur(img, k, rng.uniform(0, 180))


def jpeg_artifact(img: np.ndarray, rng: random.Random) -> np.ndarray:
    quality = rng.randint(40, 75)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img if dec is None else dec


def augment_char_fast(src_img: np.ndarray, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    img = add_margin(src_img, frac=0.08)
    img = random_affine_fast(img, rng)

    # Apply only some ops, not all ops, for speed and diversity
    if rng.random() < 0.30:
        img = random_perspective_light(img, rng)

    if rng.random() < 0.65:
        img = brightness_contrast(img, rng)

    if rng.random() < 0.30:
        img = add_noise(img, np_rng)

    # One blur path only
    r = rng.random()
    if r < 0.45:
        img = strong_blur(img, rng)
    elif r < 0.65:
        img = resize_blur(img, rng)

    if rng.random() < 0.15:
        img = jpeg_artifact(img, rng)

    return img


# =========================================================
# Dataset builders
# =========================================================
def process_normal_class(class_dir: Path, out_class_name: str) -> dict:
    images = list_images(class_dir)
    rng = random.Random(SEED + stable_seed(out_class_name))
    rng.shuffle(images)

    n = len(images)
    n_train, n_val, n_test = compute_split_counts(n)

    groups = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:n_train + n_val + n_test],
    }

    for split_name, split_paths in groups.items():
        out_dir = OUT_DIR / split_name / out_class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, src_path in enumerate(split_paths):
            dst_path = out_dir / f"{out_class_name}_{i:05d}{src_path.suffix.lower()}"
            copy_or_move(src_path, dst_path)

    return {
        "class_name": out_class_name,
        "source_name": class_dir.name,
        "total": n,
        "train": len(groups["train"]),
        "val": len(groups["val"]),
        "test": len(groups["test"]),
        "warning": None,
    }


def process_one_char_image(src_path: Path) -> dict:
    out_class_name = map_char_class_name(src_path)
    seed_base = SEED + stable_seed(src_path.stem)
    base = imread_unicode(src_path)

    # original + synthetic variants
    images = [base]
    for i in range(CHAR_TARGET_COUNT - 1):
        aug = augment_char_fast(base, seed_base + i + 1)
        images.append(aug)

    rng = random.Random(seed_base)
    rng.shuffle(images)

    n = len(images)
    n_train, n_val, n_test = compute_split_counts(n)

    groups = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:n_train + n_val + n_test],
    }

    for split_name, split_imgs in groups.items():
        out_dir = OUT_DIR / split_name / out_class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(split_imgs):
            out_path = out_dir / f"{out_class_name}_{i:05d}.jpg"
            imwrite_unicode(out_path, img)

    return {
        "class_name": out_class_name,
        "source_name": f"{SPECIAL_CHAR_FOLDER}/{src_path.name}",
        "total": n,
        "train": len(groups["train"]),
        "val": len(groups["val"]),
        "test": len(groups["test"]),
        "warning": (
            f"{out_class_name}: val/test are synthetic variants from one source image only; "
            f"they measure transform robustness, not true real-world generalization."
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
        "output_dir": str(OUT_DIR),
        "seed": SEED,
        "splits": {
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
        },
        "char_aug": {
            "target_count_per_char_class": CHAR_TARGET_COUNT,
            "blur_kernel_choices": BLUR_KERNEL_CHOICES,
            "max_workers": MAX_WORKERS,
        },
        "classes": {},
        "totals": {"train": 0, "val": 0, "test": 0, "all": 0},
        "warnings": [],
    }

    # Normal classes
    for class_dir in class_dirs:
        if class_dir.name == SPECIAL_CHAR_FOLDER:
            continue
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

    # Character classes in parallel
    char_dir = SRC_DIR / SPECIAL_CHAR_FOLDER
    if char_dir.exists() and char_dir.is_dir():
        char_images = list_images(char_dir)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(process_one_char_image, p) for p in char_images]
            for fut in as_completed(futures):
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

    print("=" * 72)
    print("Fast classification dataset prepared")
    print(f"Source: {SRC_DIR}")
    print(f"Output: {OUT_DIR}")
    print("=" * 72)

    for cls_name in sorted(summary["classes"].keys()):
        s = summary["classes"][cls_name]
        print(
            f"{cls_name:24s} | total={s['total']:4d} "
            f"train={s['train']:4d} val={s['val']:4d} test={s['test']:4d}"
        )

    print("-" * 72)
    print(
        f"TOTAL | all={summary['totals']['all']} "
        f"train={summary['totals']['train']} "
        f"val={summary['totals']['val']} "
        f"test={summary['totals']['test']}"
    )
    print("-" * 72)
    print("Done.")


if __name__ == "__main__":
    main()''