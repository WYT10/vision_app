#!/usr/bin/env python3
from pathlib import Path
import argparse
import time
import cv2


def load_warp_package(path: str):
    fs = cv2.FileStorage(path, cv2.FileStorage_READ)
    if not fs.isOpened():
        raise RuntimeError(f"failed to open warp package: {path}")

    src_w = int(fs.getNode("src_w").real())
    src_h = int(fs.getNode("src_h").real())
    warp_w = int(fs.getNode("warp_w").real())
    warp_h = int(fs.getNode("warp_h").real())
    map1 = fs.getNode("map1").mat()
    map2 = fs.getNode("map2").mat()
    valid_mask = fs.getNode("valid_mask").mat()
    family = fs.getNode("family").string()
    tag_id = int(fs.getNode("id").real())
    fs.release()

    if map1 is None or map2 is None or map1.size == 0 or map2.size == 0:
        raise RuntimeError("map1/map2 missing in warp package")

    return {
        "src_w": src_w,
        "src_h": src_h,
        "warp_w": warp_w,
        "warp_h": warp_h,
        "map1": map1,
        "map2": map2,
        "valid_mask": valid_mask,
        "family": family,
        "id": tag_id,
    }


def open_camera(device, width, height, fps, fourcc):
    try:
        dev = int(device)
    except ValueError:
        dev = device

    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open camera: {device}")

    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, actual_w, actual_h, actual_fps


def downscale_for_preview(img, max_side):
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warp", required=True, help="path to warp_package.yml.gz")
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--width", type=int, default=0, help="0 = use saved src_w")
    ap.add_argument("--height", type=int, default=0, help="0 = use saved src_h")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fourcc", default="MJPG")
    ap.add_argument("--preview-max", type=int, default=640)
    ap.add_argument("--save-dir", default="./warped_captures")
    args = ap.parse_args()

    wp = load_warp_package(args.warp)

    # Default to the saved calibration source size
    req_w = args.width if args.width > 0 else wp["src_w"]
    req_h = args.height if args.height > 0 else wp["src_h"]

    print(f"[warp] loaded {args.warp}")
    print(f"[warp] family={wp['family']} id={wp['id']}")
    print(f"[warp] saved src={wp['src_w']}x{wp['src_h']} warp={wp['warp_w']}x{wp['warp_h']}")

    cap, cam_w, cam_h, cam_fps = open_camera(
        args.device, req_w, req_h, args.fps, args.fourcc
    )
    print(f"[camera] actual mode={cam_w}x{cam_h} @ {cam_fps:.1f} fps")

    # Hard stop if camera mode does not match saved source size
    if cam_w != wp["src_w"] or cam_h != wp["src_h"]:
        cap.release()
        raise RuntimeError(
            f"live camera size {cam_w}x{cam_h} does not match saved warp source size "
            f"{wp['src_w']}x{wp['src_h']}. Re-open camera with the same mode used during calibration."
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cv2.namedWindow("warped", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("warped", min(args.preview_max, wp["warp_w"]), min(args.preview_max, wp["warp_h"]))

    save_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            print("[err] failed to read frame")
            break

        # Exact same remap style as C++ apply_warp()
        warped = cv2.remap(
            frame,
            wp["map1"],
            wp["map2"],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        show = downscale_for_preview(warped, args.preview_max)
        cv2.imshow("warped", show)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == 32:  # SPACE
            out_path = save_dir / f"warped_{time.strftime('%Y%m%d_%H%M%S')}_{save_idx:04d}.jpg"
            ok = cv2.imwrite(str(out_path), warped)
            print(f"[save] {out_path} ok={ok}")
            save_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()