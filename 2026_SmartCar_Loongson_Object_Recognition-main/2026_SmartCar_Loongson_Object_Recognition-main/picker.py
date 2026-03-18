from __future__ import annotations

import argparse
import random
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_all_images(root: Path, stem_endswith: str = "") -> list[Path]:
    images: list[Path] = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        for p in sorted(class_dir.iterdir()):
            if not (p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")):
                continue
            if stem_endswith and not p.stem.endswith(stem_endswith):
                continue
            images.append(p)
    return images


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pick N random images from the whole dataset pool."
    )
    parser.add_argument("root", help="Dataset root, e.g. dataset_cls/test")
    parser.add_argument("-n", "--num", type=int, default=5, help="Total images to pick (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stem-endswith", type=str, default="", help="Only keep files whose stem ends with this string, e.g. 0")
    parser.add_argument("--copy-to", type=str, default="", help="Optional output folder to copy selected images into")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    rng = random.Random(args.seed)
    images = list_all_images(root, stem_endswith=args.stem_endswith)

    if not images:
        print(f"No images found in: {root} with stem ending: {args.stem_endswith!r}")
        return 1

    k = min(args.num, len(images))
    chosen = rng.sample(images, k)

    print(f"Root: {root}")
    print(f"Filter stem endswith: {args.stem_endswith!r}")
    print(f"Picked {k} / {len(images)} images")
    print("-" * 80)

    if args.copy_to:
        import shutil
        out_dir = Path(args.copy_to).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    for img in chosen:
        print(img)
        if args.copy_to:
            shutil.copy2(img, out_dir / img.name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



'''
python C:\\Users\\wongy\\Downloads\\vision_app\\2026_SmartCar_Loongson_Object_Recognition-main\\2026_SmartCar_Loongson_Object_Recognition-main\\picker.py C:\\Users\\wongy\\Downloads\\vision_app\\2026_SmartCar_Loongson_Object_Recognition-main\\2026_SmartCar_Loongson_Object_Recognition-main\\generated_datasets\\dataset_cls__v3__px128__tr5__va5__te5__seed42\\test --stem-endswith 0 --copy-to C:\\Users\\wongy\\Downloads\\vision_app\\models\\5images
'''