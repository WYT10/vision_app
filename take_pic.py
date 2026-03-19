from pathlib import Path
import sys
import time
import cv2
import numpy as np


def load_warp_package(warp_path: str):
    fs = cv2.FileStorage(warp_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"failed to open warp package: {warp_path}")

    src_w = int(fs.getNode("src_w").real())
    src_h = int(fs.getNode("src_h").real())
    warp_w = int(fs.getNode("warp_w").real())
    warp_h = int(fs.getNode("warp_h").real())

    map1 = fs.getNode("map1").mat()
    map2 = fs.getNode("map2").mat()
    valid_mask = fs.getNode("valid_mask").mat()

    fs.release()

    if map1 is None or map1.size == 0:
        raise RuntimeError("map1 missing or empty")
    if map2 is None or map2.size == 0:
        raise RuntimeError("map2 missing or empty")

    return {
        "src_size": (src_w, src_h),
        "warp_size": (warp_w, warp_h),
        "map1": map1,
        "map2": map2,
        "valid_mask": valid_mask,
    }


def apply_warp(img_bgr: np.ndarray, pack: dict):
    warped = cv2.remap(
        img_bgr,
        pack["map1"],
        pack["map2"],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return warped


def fit_show(img, max_side=900):
    h, w = img.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale >= 1.0:
        return img
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def main():
    if len(sys.argv) < 3:
        print("usage:")
        print("  python warp_preview_save.py <warp_package.yml.gz> <input_image> [output_dir]")
        sys.exit(1)

    warp_path = sys.argv[1]
    img_path = sys.argv[2]
    out_dir = Path(sys.argv[3]) if len(sys.argv) >= 4 else Path("./warped_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    pack = load_warp_package(warp_path)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to load image: {img_path}")

    src_w, src_h = pack["src_size"]
    if src_w > 0 and src_h > 0 and (img.shape[1] != src_w or img.shape[0] != src_h):
        print(
            f"[warn] source image size {img.shape[1]}x{img.shape[0]} "
            f"!= saved warp src size {src_w}x{src_h}"
        )

    warped = apply_warp(img, pack)

    left = fit_show(img, 700)
    right = fit_show(warped, 700)

    # make side-by-side view
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1]
    canvas = np.full((h, w, 3), 245, dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] : left.shape[1] + right.shape[1]] = right

    cv2.putText(canvas, "original", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(
        canvas,
        "warped  [SPACE]=save  [q/ESC]=quit",
        (left.shape[1] + 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        2,
    )

    if pack["valid_mask"] is not None and pack["valid_mask"].size != 0:
        valid = pack["valid_mask"]
        valid_ratio = float(np.count_nonzero(valid)) / float(valid.size)
        print(f"[info] valid pixels: {valid_ratio * 100:.2f}%")

    print(f"[info] showing warped image, output size = {warped.shape[1]}x{warped.shape[0]}")
    cv2.imshow("warp preview", canvas)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 32:  # SPACE
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"warped_{ts}.png"
            ok = cv2.imwrite(str(out_path), warped)
            print(f"[save] {out_path}" if ok else f"[save failed] {out_path}")
        elif key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()