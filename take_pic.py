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

    if map1 is None or map2 is None:
        raise RuntimeError("map1/map2 not found in warp package")

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
    # Allow /dev/video0 or numeric camera index
    if isinstance(device, str) and device.isdigit():
        cap = cv2.VideoCapture(int(device))
    else:
        try:
            idx = int(device)
            cap = cv2.VideoCapture(idx)
        except ValueError:
            cap = cv2.VideoCapture(device)

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

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    return cap, actual_w, actual_h, actual_fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warp", required=True, help="path to warp_package.yml.gz")
    ap.add_argument("--device", default="/dev/video0", help="camera device or index")
    ap.add_argument("--width", type=int, default=0, help="requested camera width")
    ap.add_argument("--height", type=int, default=0, help="requested camera height")
    ap.add_argument("--fps", type=int, default=0, help="requested camera fps")
    ap.add_argument("--fourcc", default="MJPG", help="camera fourcc, e.g. MJPG")
    ap.add_argument("--save-dir", default="./warped_captures", help="where to save warped images")
    ap.add_argument("--show-raw", action="store_true", help="also show raw camera window")
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    wp = load_warp_package(args.warp)
    print(f"[warp] loaded: {args.warp}")
    print(f"[warp] family={wp['family']} id={wp['id']}")
    print(f"[warp] saved src size = {wp['src_w']}x{wp['src_h']}")
    print(f"[warp] warp size      = {wp['warp_w']}x{wp['warp_h']}")

    cap, cam_w, cam_h, cam_fps = open_camera(
        args.device, args.width, args.height, args.fps, args.fourcc
    )
    print(f"[camera] actual mode = {cam_w}x{cam_h} @ {cam_fps:.1f} fps")

    if cam_w != wp["src_w"] or cam_h != wp["src_h"]:
        print("[warn] live camera size does not match saved warp source size")
        print(f"       live={cam_w}x{cam_h} saved={wp['src_w']}x{wp['src_h']}")
        print("       warped output may be wrong until camera mode matches calibration")

    cv2.namedWindow("warped", cv2.WINDOW_NORMAL)
    if args.show_raw:
        cv2.namedWindow("raw", cv2.WINDOW_NORMAL)

    save_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[err] failed to read frame")
            break

        warped = cv2.remap(
            frame,
            wp["map1"],
            wp["map2"],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        # Show clean warped image (no text/ROI overlays)
        cv2.imshow("warped", warped)
        if args.show_raw:
            cv2.imshow("raw", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == 32:  # SPACE
            out_path = save_dir / f"warped_{time.strftime('%Y%m%d_%H%M%S')}_{save_idx:04d}.jpg"
            cv2.imwrite(str(out_path), warped)
            print(f"[save] {out_path}")
            save_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()