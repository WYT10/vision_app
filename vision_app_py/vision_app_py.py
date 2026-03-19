#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import ncnn  # optional
except Exception:
    ncnn = None

@dataclass
class RoiRect:
    x: int = 0
    y: int = 0
    w: int = 64
    h: int = 64

    def clamp(self, W: int, H: int) -> "RoiRect":
        x = max(0, min(self.x, max(0, W - 1)))
        y = max(0, min(self.y, max(0, H - 1)))
        w = max(1, min(self.w, max(1, W - x)))
        h = max(1, min(self.h, max(1, H - y)))
        return RoiRect(x, y, w, h)

@dataclass
class WarpConfig:
    family: str = "auto"
    target_id: int = 0
    require_target_id: bool = False
    src_width: int = 0
    src_height: int = 0
    warp_width: int = 384
    warp_height: int = 384
    target_tag_px: int = 128
    H: list = None
    image_roi: RoiRect = None
    red_roi: RoiRect = None

    def __post_init__(self):
        if self.H is None:
            self.H = np.eye(3, dtype=np.float32).tolist()
        if self.image_roi is None:
            self.image_roi = RoiRect(128, 128, 128, 128)
        if self.red_roi is None:
            self.red_roi = RoiRect(24, 24, 64, 64)

    @staticmethod
    def from_dict(d: dict) -> "WarpConfig":
        return WarpConfig(
            family=d.get("family", "auto"),
            target_id=int(d.get("target_id", 0)),
            require_target_id=bool(d.get("require_target_id", False)),
            src_width=int(d.get("src_width", 0)),
            src_height=int(d.get("src_height", 0)),
            warp_width=int(d.get("warp_width", 384)),
            warp_height=int(d.get("warp_height", 384)),
            target_tag_px=int(d.get("target_tag_px", 128)),
            H=d.get("H", np.eye(3, dtype=np.float32).tolist()),
            image_roi=RoiRect(**d.get("image_roi", {})),
            red_roi=RoiRect(**d.get("red_roi", {})),
        )

    def to_dict(self) -> dict:
        out = asdict(self)
        out["image_roi"] = asdict(self.image_roi)
        out["red_roi"] = asdict(self.red_roi)
        return out

COMMON_PROBE = [
    (160, 120, "MJPG", 120),
    (320, 240, "MJPG", 120),
    (640, 480, "MJPG", 60),
    (1280, 720, "MJPG", 30),
    (640, 480, "YUYV", 30),
]

DICTS = {
    "16": cv2.aruco.DICT_APRILTAG_16h5,
    "25": cv2.aruco.DICT_APRILTAG_25h9,
    "36": cv2.aruco.DICT_APRILTAG_36h11,
}

def fourcc_code(tag: str) -> int:
    tag = (tag or "MJPG").ljust(4)[:4]
    return cv2.VideoWriter_fourcc(*tag)

def open_camera(device: str, width: int, height: int, fps: int, fourcc: str, buffer_size: int = 1):
    device_str = str(device)
    if device_str.isdigit() and os.name == "nt":
        # On Windows prefer DirectShow for USB cameras.
        cap = cv2.VideoCapture(int(device_str) + cv2.CAP_DSHOW)
    elif device_str.isdigit():
        cap = cv2.VideoCapture(int(device_str))
    else:
        cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {device}")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc_code(fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    except Exception:
        pass
    return cap

def grab_latest(cap, drain_grabs: int = 1):
    frame = None
    for _ in range(max(1, drain_grabs)):
        ok, frame = cap.read()
        if not ok:
            return None
    return frame

def _make_detectors(family: str):
    params = cv2.aruco.DetectorParameters()
    try:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    except Exception:
        pass
    fams = ["16", "25", "36"] if family == "auto" else [family]
    detectors = []
    for f in fams:
        detectors.append((f, cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTS[f]), params)))
    return detectors

def detect_best_tag(image_bgr: np.ndarray, family: str = "auto", target_id: int = 0, require_target_id: bool = False):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    best = None
    for fam, det in _make_detectors(family):
        corners, ids, _ = det.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            continue
        ids = ids.flatten().tolist()
        for c, mid in zip(corners, ids):
            if require_target_id and int(mid) != int(target_id):
                continue
            pts = c.reshape(4, 2).astype(np.float32)
            area = cv2.contourArea(pts)
            cand = {"family": fam, "id": int(mid), "corners": pts, "area": float(abs(area))}
            if best is None or cand["area"] > best["area"]:
                best = cand
    return best

def build_centered_homography(src_corners: np.ndarray, warp_w: int, warp_h: int, target_tag_px: int) -> np.ndarray:
    cx, cy = warp_w / 2.0, warp_h / 2.0
    s = target_tag_px / 2.0
    dst = np.array([
        [cx - s, cy - s],
        [cx + s, cy - s],
        [cx + s, cy + s],
        [cx - s, cy + s],
    ], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_corners.astype(np.float32), dst)
    return H

def apply_warp(frame: np.ndarray, cfg: WarpConfig) -> np.ndarray:
    H = np.asarray(cfg.H, dtype=np.float32)
    return cv2.warpPerspective(frame, H, (cfg.warp_width, cfg.warp_height))

def crop_roi(image: np.ndarray, roi: RoiRect) -> np.ndarray:
    r = roi.clamp(image.shape[1], image.shape[0])
    return image[r.y:r.y + r.h, r.x:r.x + r.w].copy()

def compute_red_ratio(image_bgr: np.ndarray, h1_low=0, h1_high=10, h2_low=170, h2_high=180, s_min=80, v_min=60):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (h1_low, s_min, v_min), (h1_high, 255, 255))
    m2 = cv2.inRange(hsv, (h2_low, s_min, v_min), (h2_high, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    red_pixels = int(np.count_nonzero(mask))
    total = int(mask.size)
    ratio = 0.0 if total <= 0 else red_pixels / total
    return ratio, mask

def parse_labels(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]

def preprocess_image(image_bgr: np.ndarray, size: int, mode: str = "crop", mean=(0.0, 0.0, 0.0), norm=(1/255.0, 1/255.0, 1/255.0)) -> np.ndarray:
    img = image_bgr
    H, W = img.shape[:2]
    if mode == "crop":
        s = min(H, W)
        y0 = (H - s) // 2
        x0 = (W - s) // 2
        img = img[y0:y0+s, x0:x0+s]
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    elif mode == "stretch":
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    elif mode == "letterbox":
        scale = min(size / W, size / H)
        nw, nh = max(1, int(W * scale)), max(1, int(H * scale))
        rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        y0 = (size - nh) // 2
        x0 = (size - nw) // 2
        canvas[y0:y0+nh, x0:x0+nw] = rs
        img = canvas
    else:
        raise ValueError(f"Unknown preprocess mode: {mode}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean = np.array(mean, dtype=np.float32).reshape(1,1,3)
    norm = np.array(norm, dtype=np.float32).reshape(1,1,3)
    img = (img - mean) * norm
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return img

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.maximum(1e-12, np.sum(e))

class OnnxClassifier:
    def __init__(self, model_path: str, labels_path: str, size: int = 128, prep: str = "crop", mean=(0,0,0), norm=(1/255,1/255,1/255), threads: int = 1):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed")
        self.labels = parse_labels(labels_path)
        self.size = size
        self.prep = prep
        self.mean = mean
        self.norm = norm
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, int(threads))
        self.sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def __call__(self, image_bgr: np.ndarray, topk: int = 5):
        x = preprocess_image(image_bgr, self.size, self.prep, self.mean, self.norm)
        y = self.sess.run([self.output_name], {self.input_name: x})[0].reshape(-1)
        probs = softmax(y)
        idx = np.argsort(-probs)[:topk]
        return [(int(i), self.labels[int(i)] if int(i) < len(self.labels) else str(int(i)), float(probs[int(i)])) for i in idx]

class NcnnClassifier:
    def __init__(self, param_path: str, bin_path: str, labels_path: str, size: int = 128, prep: str = "crop", mean=(0,0,0), norm=(1/255,1/255,1/255), threads: int = 4):
        if ncnn is None:
            raise RuntimeError("python ncnn module is not installed")
        self.labels = parse_labels(labels_path)
        self.size = size
        self.prep = prep
        self.mean = mean
        self.norm = norm
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = max(1, int(threads))
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        self.input_name, self.output_name = infer_ncnn_io_names(param_path)

    def __call__(self, image_bgr: np.ndarray, topk: int = 5):
        # Python ncnn path simplified: use pixels path instead of manual NCHW
        img = image_bgr
        H, W = img.shape[:2]
        if self.prep == "crop":
            s = min(H, W)
            y0 = (H - s) // 2
            x0 = (W - s) // 2
            img = img[y0:y0+s, x0:x0+s]
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        elif self.prep == "stretch":
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        elif self.prep == "letterbox":
            scale = min(self.size / W, self.size / H)
            nw, nh = max(1, int(W * scale)), max(1, int(H * scale))
            rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            y0 = (self.size - nh) // 2
            x0 = (self.size - nw) // 2
            canvas[y0:y0+nh, x0:x0+nw] = rs
            img = canvas
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mat = ncnn.Mat.from_pixels(img_rgb, ncnn.Mat.PixelType.PIXEL_RGB, self.size, self.size)
        mat.substract_mean_normalize(self.mean, self.norm)
        ex = self.net.create_extractor()
        ex.input(self.input_name, mat)
        ret, out = ex.extract(self.output_name)
        if ret != 0:
            raise RuntimeError(f"ncnn extract failed: {ret}, output={self.output_name}")
        y = np.array(out, dtype=np.float32).reshape(-1)
        probs = softmax(y)
        idx = np.argsort(-probs)[:topk]
        return [(int(i), self.labels[int(i)] if int(i) < len(self.labels) else str(int(i)), float(probs[int(i)])) for i in idx]

def infer_ncnn_io_names(param_path: str) -> Tuple[str, str]:
    with open(param_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    layer_lines = []
    for ln in lines:
        if ln == "7767517":
            continue
        toks = ln.split()
        if len(toks) >= 4 and toks[2].isdigit() and toks[3].isdigit():
            layer_lines.append(toks)
    produced = []
    consumed = set()
    input_name = None
    for toks in layer_lines:
        ltype = toks[0]
        bottom_count = int(toks[2]); top_count = int(toks[3]); pos = 4
        bottoms = toks[pos:pos+bottom_count]; pos += bottom_count
        tops = toks[pos:pos+top_count]
        if ltype == "Input" and tops:
            input_name = tops[0]
        for b in bottoms: consumed.add(b)
        for t in tops: produced.append(t)
    if input_name is None:
        input_name = "in0"
    output_name = None
    for name in reversed(produced):
        if name not in consumed:
            output_name = name
            break
    if output_name is None:
        output_name = produced[-1] if produced else "out0"
    return input_name, output_name

def save_cfg(path: str, cfg: WarpConfig):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

def save_warped_image(save_dir: str, frame_idx: int, warped: np.ndarray):
    if not save_dir:
        return
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(Path(save_dir) / f"warped_{frame_idx:06d}.jpg"), warped)

def load_cfg(path: str) -> WarpConfig:
    return WarpConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

def draw_text(img, text, org, color=(0,120,0), scale=0.5, thick=1, enabled=True):
    if enabled:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_rois(img, cfg: WarpConfig):
    cv2.rectangle(img, (cfg.red_roi.x, cfg.red_roi.y), (cfg.red_roi.x+cfg.red_roi.w, cfg.red_roi.y+cfg.red_roi.h), (0,0,255), 2)
    cv2.putText(img, "red_roi", (cfg.red_roi.x, max(12, cfg.red_roi.y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.rectangle(img, (cfg.image_roi.x, cfg.image_roi.y), (cfg.image_roi.x+cfg.image_roi.w, cfg.image_roi.y+cfg.image_roi.h), (255,0,0), 2)
    cv2.putText(img, "image_roi", (cfg.image_roi.x, max(12, cfg.image_roi.y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

def mode_probe(args):
    rows = []
    tests = COMMON_PROBE if args.scan_common else [(args.width, args.height, args.fourcc, args.fps)]
    for w,h,fcc,fps in tests:
        rec = {"width": w, "height": h, "fourcc": fcc, "fps_req": fps, "opened": False}
        try:
            cap = open_camera(args.device, w, h, fps, fcc, args.buffer_size)
            rec["opened"] = True
            time.sleep(args.warmup)
            n = 0
            t0 = time.time()
            preview_frame = None
            while time.time() - t0 < args.duration:
                fr = grab_latest(cap, args.drain_grabs)
                if fr is None:
                    break
                preview_frame = fr
                n += 1
                if args.preview:
                    cv2.imshow("vision_app_probe", fr)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                        break
            dt = max(1e-6, time.time() - t0)
            rec["fps_measured"] = n / dt
            if preview_frame is not None:
                rec["actual_width"] = int(preview_frame.shape[1]); rec["actual_height"] = int(preview_frame.shape[0])
            cap.release()
        except Exception as e:
            rec["error"] = str(e)
        rows.append(rec)
    if args.preview:
        cv2.destroyAllWindows()
    print("\n=== probe results ===")
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    if args.report:
        Path(args.report).write_text(json.dumps({"results": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

def mode_calibrate(args):
    cap = open_camera(args.device, args.width, args.height, args.fps, args.fourcc, args.buffer_size)
    cfg = WarpConfig(
        family=args.tag_family, target_id=args.target_id, require_target_id=bool(args.require_target_id),
        src_width=args.width, src_height=args.height, warp_width=args.warp_width, warp_height=args.warp_height,
        target_tag_px=args.target_tag_px,
        image_roi=RoiRect(args.image_roi_x, args.image_roi_y, args.image_roi_w, args.image_roi_h),
        red_roi=RoiRect(args.red_roi_x, args.red_roi_y, args.red_roi_w, args.red_roi_h),
    )
    selected, move_step, size_step, locked, last_warp = "image", 4, 4, False, None
    print("\n=== calibrate controls ===")
    print("SPACE/ENTER lock current tag | u unlock | 1/2 select roi | wasd move | i/k h-/+ | j/l w-/+ | [/] move step | ,/. size step | p save | q quit")
    while True:
        frame = grab_latest(cap, args.drain_grabs)
        if frame is None:
            raise RuntimeError("Failed to read frame")
        det = detect_best_tag(frame, cfg.family, cfg.target_id, cfg.require_target_id)
        cam_show = frame.copy()
        if det is not None and not locked:
            pts = det["corners"].astype(np.int32)
            cv2.polylines(cam_show, [pts], True, (0,255,0), 2)
            H = build_centered_homography(det["corners"], cfg.warp_width, cfg.warp_height, cfg.target_tag_px)
            cfg.H = H.tolist()
            last_warp = apply_warp(frame, cfg)
        elif locked:
            last_warp = apply_warp(frame, cfg)
        warp_show = np.zeros((cfg.warp_height, cfg.warp_width, 3), dtype=np.uint8) if last_warp is None else last_warp.copy()
        draw_rois(warp_show, cfg)
        draw_text(warp_show, f'{"LOCK" if locked else "SEARCH"} warp={cfg.warp_width}x{cfg.warp_height} tag_px={cfg.target_tag_px}', (12, 24), enabled=bool(args.draw_text))
        cv2.imshow("vision_app_camera", cam_show)
        cv2.imshow("vision_app_warp", warp_show)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key in (13, 32):
            if det is not None:
                H = build_centered_homography(det["corners"], cfg.warp_width, cfg.warp_height, cfg.target_tag_px)
                cfg.H = H.tolist(); locked = True
                print(f"[lock] family={det['family']} id={det['id']}")
        elif key == ord('u'):
            locked = False; print("[unlock]")
        elif key == ord('1'):
            selected = "red"; print("[roi] selected red_roi")
        elif key == ord('2'):
            selected = "image"; print("[roi] selected image_roi")
        else:
            roi = cfg.red_roi if selected == "red" else cfg.image_roi
            if key == ord('w'): roi.y -= move_step
            elif key == ord('s'): roi.y += move_step
            elif key == ord('a'): roi.x -= move_step
            elif key == ord('d'): roi.x += move_step
            elif key == ord('i'): roi.h = max(1, roi.h - size_step)
            elif key == ord('k'): roi.h += size_step
            elif key == ord('j'): roi.w = max(1, roi.w - size_step)
            elif key == ord('l'): roi.w += size_step
            elif key == ord('['): move_step = max(1, move_step - 1)
            elif key == ord(']'): move_step += 1
            elif key == ord(','): size_step = max(1, size_step - 1)
            elif key == ord('.'): size_step += 1
            elif key == ord('r'):
                cfg.red_roi = RoiRect(24, 24, 64, 64); cfg.image_roi = RoiRect(128, 128, 128, 128)
            elif key in (ord('p'), ord('y'), ord('o')):
                save_cfg(args.save_config, cfg); print(f"[save] -> {args.save_config}")
            if selected == "red": cfg.red_roi = roi.clamp(cfg.warp_width, cfg.warp_height)
            else: cfg.image_roi = roi.clamp(cfg.warp_width, cfg.warp_height)
    cap.release(); cv2.destroyAllWindows()

def build_classifier(args):
    if not args.model_enable:
        return None
    if args.model_backend == "onnx":
        return OnnxClassifier(args.model_onnx_path, args.model_labels_path, size=args.model_input_width, prep=args.model_preprocess, threads=args.model_threads)
    if args.model_backend == "ncnn":
        return NcnnClassifier(args.model_ncnn_param_path, args.model_ncnn_bin_path, args.model_labels_path, size=args.model_input_width, prep=args.model_preprocess, threads=args.model_threads)
    raise ValueError(f"Unknown model backend: {args.model_backend}")

def mode_deploy(args):
    cfg = load_cfg(args.load_config)
    cap = open_camera(args.device, args.width, args.height, args.fps, args.fourcc, args.buffer_size)
    clf = build_classifier(args)
    last_pred_t, frame_idx, pred = 0.0, 0, None
    pred_every = 0.0 if args.model_max_hz <= 0 else 1.0 / args.model_max_hz
    if args.save_image_roi_dir: Path(args.save_image_roi_dir).mkdir(parents=True, exist_ok=True)
    if args.save_red_roi_dir: Path(args.save_red_roi_dir).mkdir(parents=True, exist_ok=True)
    if args.save_warped_dir: Path(args.save_warped_dir).mkdir(parents=True, exist_ok=True)
    if args.width != cfg.src_width or args.height != cfg.src_height:
        print(f"[warn] live camera size {args.width}x{args.height} does not match saved config source size {cfg.src_width}x{cfg.src_height}")
    while True:
        t0 = time.perf_counter()
        frame = grab_latest(cap, args.drain_grabs)
        if frame is None: raise RuntimeError("Failed to read frame")
        t1 = time.perf_counter()
        warped = apply_warp(frame, cfg)
        t2 = time.perf_counter()
        red_roi = crop_roi(warped, cfg.red_roi)
        image_roi = crop_roi(warped, cfg.image_roi)
        red_ratio, red_mask = compute_red_ratio(red_roi, args.red_h1_low, args.red_h1_high, args.red_h2_low, args.red_h2_high, args.red_s_min, args.red_v_min)
        t3 = time.perf_counter()
        if args.save_every_n > 0 and frame_idx % args.save_every_n == 0:
            # Save raw warped frame before any UI drawing overlays are added.
            if args.save_warped_dir: save_warped_image(args.save_warped_dir, frame_idx, warped)
            if args.save_image_roi_dir and args.run_image_roi: cv2.imwrite(str(Path(args.save_image_roi_dir) / f"image_roi_{frame_idx:06d}.jpg"), image_roi)
            if args.save_red_roi_dir and args.run_red: cv2.imwrite(str(Path(args.save_red_roi_dir) / f"red_roi_{frame_idx:06d}.jpg"), red_roi)
        run_model_now = args.run_model and clf is not None and ((args.model_stride <= 1) or (frame_idx % args.model_stride == 0)) and (pred_every <= 0 or (time.perf_counter() - last_pred_t) >= pred_every)
        model_ms = 0.0
        if run_model_now:
            mt0 = time.perf_counter(); pred = clf(image_roi, topk=args.model_topk); model_ms = (time.perf_counter()-mt0)*1000.0; last_pred_t = time.perf_counter()
        if not args.headless:
            cam_show = frame.copy(); warp_show = warped.copy()
            cv2.rectangle(warp_show, (cfg.red_roi.x, cfg.red_roi.y), (cfg.red_roi.x+cfg.red_roi.w, cfg.red_roi.y+cfg.red_roi.h), (0,0,255), 2)
            cv2.rectangle(warp_show, (cfg.image_roi.x, cfg.image_roi.y), (cfg.image_roi.x+cfg.image_roi.w, cfg.image_roi.y+cfg.image_roi.h), (255,0,0), 2)
            if args.draw_text:
                draw_text(warp_show, f"red_ratio={red_ratio:.3f}", (12, 24))
                draw_text(warp_show, f"cap={(t1-t0)*1000:.1f} warp={(t2-t1)*1000:.1f} roi={(t3-t2)*1000:.1f} model={model_ms:.1f}", (12, 48))
                draw_text(warp_show, f"hz<={args.model_max_hz:.2f} stride={args.model_stride}", (12, 72))
                if pred: idx, label, score = pred[0]; draw_text(warp_show, f"{args.model_backend} {label} {score:.4f}", (12, 96))
            cv2.imshow("vision_app_camera", cam_show); cv2.imshow("vision_app_warp", warp_show)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')): break
        else:
            if pred:
                idx, label, score = pred[0]
                print(f"frame={frame_idx} red_ratio={red_ratio:.3f} label={label} score={score:.4f} model_ms={model_ms:.2f}")
        frame_idx += 1
    cap.release(); cv2.destroyAllWindows()

def make_parser():
    p = argparse.ArgumentParser(description="All-in-one Python vision app with probe / calibrate / deploy")
    p.add_argument("--mode", required=True, choices=["probe", "calibrate", "deploy"])
    p.add_argument("--device", default="1" if os.name == "nt" else "/dev/video0")
    p.add_argument("--width", type=int, default=160)
    p.add_argument("--height", type=int, default=120)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--fourcc", default="MJPG")
    p.add_argument("--buffer-size", type=int, default=1)
    p.add_argument("--drain-grabs", type=int, default=1)
    p.add_argument("--headless", type=int, default=0)
    p.add_argument("--draw-text", type=int, default=1)
    p.add_argument("--scan-common", type=int, default=0)
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--warmup", type=float, default=0.5)
    p.add_argument("--preview", type=int, default=1)
    p.add_argument("--report", default="")
    p.add_argument("--tag-family", default="auto", choices=["auto", "16", "25", "36"])
    p.add_argument("--target-id", type=int, default=0)
    p.add_argument("--require-target-id", type=int, default=0)
    p.add_argument("--warp-width", type=int, default=384)
    p.add_argument("--warp-height", type=int, default=384)
    p.add_argument("--target-tag-px", type=int, default=128)
    p.add_argument("--save-config", default="./report/vision_app_calibration.json")
    p.add_argument("--load-config", default="./report/vision_app_calibration.json")
    p.add_argument("--image-roi-x", type=int, default=128)
    p.add_argument("--image-roi-y", type=int, default=128)
    p.add_argument("--image-roi-w", type=int, default=128)
    p.add_argument("--image-roi-h", type=int, default=128)
    p.add_argument("--red-roi-x", type=int, default=24)
    p.add_argument("--red-roi-y", type=int, default=24)
    p.add_argument("--red-roi-w", type=int, default=64)
    p.add_argument("--red-roi-h", type=int, default=64)
    p.add_argument("--red-h1-low", type=int, default=0)
    p.add_argument("--red-h1-high", type=int, default=10)
    p.add_argument("--red-h2-low", type=int, default=170)
    p.add_argument("--red-h2-high", type=int, default=180)
    p.add_argument("--red-s-min", type=int, default=80)
    p.add_argument("--red-v-min", type=int, default=60)
    p.add_argument("--run-red", type=int, default=1)
    p.add_argument("--run-image-roi", type=int, default=1)
    p.add_argument("--run-model", type=int, default=1)
    p.add_argument("--save-image-roi-dir", default="")
    p.add_argument("--save-red-roi-dir", default="")
    p.add_argument("--save-warped-dir", default="")
    p.add_argument("--save-every-n", type=int, default=0)
    p.add_argument("--model-enable", type=int, default=0)
    p.add_argument("--model-backend", default="ncnn", choices=["onnx", "ncnn"])
    p.add_argument("--model-onnx-path", default="")
    p.add_argument("--model-ncnn-param-path", default="")
    p.add_argument("--model-ncnn-bin-path", default="")
    p.add_argument("--model-labels-path", default="")
    p.add_argument("--model-input-width", type=int, default=128)
    p.add_argument("--model-input-height", type=int, default=128)
    p.add_argument("--model-preprocess", default="crop", choices=["crop", "stretch", "letterbox"])
    p.add_argument("--model-threads", type=int, default=4)
    p.add_argument("--model-stride", type=int, default=1)
    p.add_argument("--model-max-hz", type=float, default=5.0)
    p.add_argument("--model-topk", type=int, default=5)
    return p

def main():
    args = make_parser().parse_args()
    try:
        if args.mode == "probe": mode_probe(args)
        elif args.mode == "calibrate": mode_calibrate(args)
        elif args.mode == "deploy": mode_deploy(args)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"Error: {e}", file=None)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
