#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from PIL import Image, ImageOps

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


# ---------- stimulus format ----------
STIM_IMAGE_W = 240
STIM_IMAGE_H = 240
STIM_RED_W = 240
STIM_RED_H = 100
STIM_GAP_H = 20
STIM_BG = (255, 255, 255)
STIM_RED = (255, 0, 0)

# calibration tag content shown on /calibrate_tag
APRILTAG_FAMILY = "36h11"
APRILTAG_ID = 0

# collect policy (kept as constants so server args stay minimal)
LOW_CONF_THRESHOLD = 0.65
MAX_SAVED_PER_CLASS = 200
MAX_SAVED_TOTAL = 3000
DISK_CAP_MB = 2048
AUTO_ADVANCE = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
API_VERSION = 2

# Built-in bilingual label aliases. These are canonicalized to the English model-side names.
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
class Item:
    label: str
    path: str
    relpath: str
    name: str


@dataclass
class CurrentTrial:
    session: str
    trial_id: str
    label: str
    relpath: str
    image_name: str
    item_index: int
    mode: str


def safe_name(s: str) -> str:
    out = []
    for ch in s:
        out.append(ch if (ch.isalnum() or ch in ("_", "-", ".")) else "_")
    return "".join(out)


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_demo_stimulus(src_path: Path) -> bytes:
    src = Image.open(src_path).convert("RGB")
    obj = ImageOps.pad(
        src,
        (STIM_IMAGE_W, STIM_IMAGE_H),
        method=Image.Resampling.BILINEAR,
        color=STIM_BG,
        centering=(0.5, 0.5),
    )

    total_w = max(STIM_IMAGE_W, STIM_RED_W)
    total_h = STIM_IMAGE_H + STIM_GAP_H + STIM_RED_H
    canvas = Image.new("RGB", (total_w, total_h), STIM_BG)

    canvas.paste(obj, ((total_w - STIM_IMAGE_W) // 2, 0))
    red = Image.new("RGB", (STIM_RED_W, STIM_RED_H), STIM_RED)
    canvas.paste(red, ((total_w - STIM_RED_W) // 2, STIM_IMAGE_H + STIM_GAP_H))
    return image_to_png_bytes(canvas)


def build_apriltag_core(size: int = STIM_IMAGE_W) -> Image.Image:
    if cv2 is None or not hasattr(cv2, "aruco"):
        img = Image.new("L", (size, size), 255)
        step = max(8, size // 12)
        for i in range(0, size, step):
            for j in range(0, size, step):
                if ((i // step) + (j // step)) % 2 == 0:
                    for y in range(j, min(size, j + step)):
                        for x in range(i, min(size, i + step)):
                            img.putpixel((x, y), 0)
        return img.convert("RGB")

    family_map = {
        "16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "36h11": cv2.aruco.DICT_APRILTAG_36h11,
    }
    dict_id = family_map.get(APRILTAG_FAMILY, cv2.aruco.DICT_APRILTAG_36h11)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(dictionary, APRILTAG_ID, size)
    else:
        marker = cv2.aruco.drawMarker(dictionary, APRILTAG_ID, size)

    return Image.fromarray(marker).convert("RGB")


def build_tag_stimulus() -> bytes:
    tag_img = ImageOps.pad(
        build_apriltag_core(STIM_IMAGE_W),
        (STIM_IMAGE_W, STIM_IMAGE_H),
        method=Image.Resampling.NEAREST,
        color=STIM_BG,
        centering=(0.5, 0.5),
    )

    total_w = max(STIM_IMAGE_W, STIM_RED_W)
    total_h = STIM_IMAGE_H + STIM_GAP_H + STIM_RED_H
    canvas = Image.new("RGB", (total_w, total_h), STIM_BG)

    canvas.paste(tag_img, ((total_w - STIM_IMAGE_W) // 2, 0))
    red = Image.new("RGB", (STIM_RED_W, STIM_RED_H), STIM_RED)
    canvas.paste(red, ((total_w - STIM_RED_W) // 2, STIM_IMAGE_H + STIM_GAP_H))
    return image_to_png_bytes(canvas)


def display_html() -> bytes:
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0,user-scalable=no">
<title>vision_app display</title>
<style>
html,body{margin:0;padding:0;background:#fff;width:100%;height:100%;overflow:hidden}
body{display:flex;align-items:center;justify-content:center}
img{max-width:95vw;max-height:95vh;object-fit:contain}
.info{position:fixed;left:8px;bottom:8px;font-family:sans-serif;font-size:12px;color:#555;background:rgba(255,255,255,.8);padding:4px 6px;border-radius:6px}
</style>
</head>
<body>
  <img id="stim" src="/img" alt="stimulus">
  <div class="info" id="info">connecting...</div>
<script>
async function refreshMeta(){
  try{
    const r = await fetch('/api/current?t=' + Date.now(), {cache:'no-store'});
    const j = await r.json();
    document.getElementById('info').textContent =
      'trial=' + j.trial_id + ' label=' + j.label + ' active=' + j.active;
  }catch(e){
    document.getElementById('info').textContent = 'server not ready';
  }
}
function refreshImg(){ document.getElementById('stim').src = '/img?t=' + Date.now(); }
setInterval(refreshImg, 500);
setInterval(refreshMeta, 500);
refreshImg();
refreshMeta();
</script>
</body>
</html>"""
    return html.encode("utf-8")


def tag_html() -> bytes:
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0,user-scalable=no">
<title>vision_app calibrate tag</title>
<style>
html,body{margin:0;padding:0;background:#fff;width:100%;height:100%;overflow:hidden}
body{display:flex;align-items:center;justify-content:center}
img{max-width:95vw;max-height:95vh;object-fit:contain}
.info{position:fixed;left:8px;bottom:8px;font-family:sans-serif;font-size:12px;color:#555;background:rgba(255,255,255,.8);padding:4px 6px;border-radius:6px}
</style>
</head>
<body>
  <img id="stim" src="/tag" alt="apriltag stimulus">
  <div class="info">apriltag stimulus · same format as demo</div>
<script>
function refreshImg(){ document.getElementById('stim').src = '/tag?t=' + Date.now(); }
setInterval(refreshImg, 1000);
refreshImg();
</script>
</body>
</html>"""
    return html.encode("utf-8")


class LabelNormalizer:
    def __init__(self, dataset_root: Path, output_root: Path) -> None:
        self.dataset_root = dataset_root
        self.output_root = output_root
        self.aliases = self._load_aliases()

    def _load_aliases(self) -> Dict[str, str]:
        merged: Dict[str, str] = dict(DEFAULT_LABEL_ALIASES)
        candidates = [
            self.output_root / "label_aliases.json",
            self.dataset_root / "label_aliases.json",
        ]
        for p in candidates:
            if not p.exists():
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    for k, v in data.items():
                        merged[str(k)] = str(v)
            except Exception:
                pass
        return merged

    def canonical(self, label: str) -> str:
        s = (label or "").strip()
        if s in self.aliases:
            return self.aliases[s].strip().lower()

        s2 = s.replace(" ", "_").replace("-", "_").strip("_")
        if s2 in self.aliases:
            return self.aliases[s2].strip().lower()

        # Fallback heuristic: compare by class prefix if no exact alias exists.
        m = re.match(r"^([A-Za-z][A-Za-z0-9]*)[-_\s].*$", s)
        if m:
            return m.group(1).lower()

        return s2.lower()

    def compare(self, expected: str, predicted: str) -> Tuple[bool, str, str]:
        e = self.canonical(expected)
        p = self.canonical(predicted)
        return (e == p), e, p


class SessionState:
    def __init__(self, dataset_root: Path, output_root: Path, mode: str) -> None:
        self.dataset_root = dataset_root.resolve()
        self.output_root = output_root.resolve()
        self.mode = mode

        self.sessions_root = self.output_root / "sessions"
        self.workspaces_root = self.output_root / "workspaces"
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self.workspaces_root.mkdir(parents=True, exist_ok=True)

        self.session = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = self.sessions_root / self.session
        self.workspace_dir = self.workspaces_root / self.session
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.label_normalizer = LabelNormalizer(self.dataset_root, self.output_root)
        self.items = self._scan_items()
        if not self.items:
            raise RuntimeError(f"No images found under dataset root: {self.dataset_root}")

        self.lock = threading.Lock()
        self.current_index = 0
        self.current_trial = self._new_trial_locked()
        self.results_path = self.run_dir / "results.jsonl"
        self.session_path = self.run_dir / "session.json"

        self.class_saved_counts: Dict[str, int] = {}
        self.total_saved = 0
        self.last_post_summary = "idle"

        self._write_session_json()

    def _scan_items(self) -> List[Item]:
        items: List[Item] = []
        for class_dir in sorted(self.dataset_root.iterdir()):
            if not class_dir.is_dir() or class_dir.name.startswith("."):
                continue
            label = class_dir.name
            for path in sorted(class_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in IMG_EXTS:
                    continue
                items.append(
                    Item(
                        label=label,
                        path=str(path.resolve()),
                        relpath=str(path.resolve().relative_to(self.dataset_root)),
                        name=path.name,
                    )
                )
        return items

    def _new_trial_locked(self) -> CurrentTrial:
        item = self.items[self.current_index]
        return CurrentTrial(
            session=self.session,
            trial_id=f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
            label=item.label,
            relpath=item.relpath,
            image_name=item.name,
            item_index=self.current_index,
            mode=self.mode,
        )

    def _write_session_json(self) -> None:
        payload = {
            "session": self.session,
            "mode": self.mode,
            "dataset_root": str(self.dataset_root),
            "output_root": str(self.output_root),
            "run_dir": str(self.run_dir),
            "workspace_dir": str(self.workspace_dir),
            "items": len(self.items),
            "stimulus": {
                "image_w": STIM_IMAGE_W,
                "image_h": STIM_IMAGE_H,
                "red_w": STIM_RED_W,
                "red_h": STIM_RED_H,
                "gap_h": STIM_GAP_H,
            },
            "calibrate_tag": {
                "family": APRILTAG_FAMILY,
                "id": APRILTAG_ID,
            },
            "label_aliases_path_output": str(self.output_root / "label_aliases.json"),
            "label_aliases_path_dataset": str(self.dataset_root / "label_aliases.json"),
            "built_in_aliases_count": len(DEFAULT_LABEL_ALIASES),
            "created_at": datetime.now().isoformat(),
        }
        self.session_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def get_current_item_locked(self) -> Item:
        return self.items[self.current_index]

    def current_trial_payload(self) -> dict:
        t = self.current_trial
        return {
            "ok": True,
            "active": True,
            "session": t.session,
            "trial_id": t.trial_id,
            "label": t.label,
            "image_name": t.image_name,
            "image_url": "/img",
            "version": API_VERSION,
            "item_index": t.item_index,
            "mode": t.mode,
        }

    def current_payload(self) -> dict:
        with self.lock:
            return self.current_trial_payload()

    def advance(self) -> dict:
        with self.lock:
            self.current_index = (self.current_index + 1) % len(self.items)
            self.current_trial = self._new_trial_locked()
            return self.current_trial_payload()

    def current_stimulus_png(self) -> bytes:
        with self.lock:
            return build_demo_stimulus(Path(self.get_current_item_locked().path))

    def append_result(self, payload: dict) -> Tuple[bool, Optional[str], str]:
        with self.lock:
            current_item = self.get_current_item_locked()
            expected = str(payload.get("expected_label", current_item.label))
            predicted = str(payload.get("predicted_label", ""))
            normalized_match, expected_canonical, predicted_canonical = self.label_normalizer.compare(expected, predicted)

            payload["server_time"] = time.time()
            payload["server_session"] = self.session
            payload["item_index"] = self.current_index
            payload["current_relpath"] = current_item.relpath
            payload["current_label"] = current_item.label
            payload["expected_canonical"] = expected_canonical
            payload["predicted_canonical"] = predicted_canonical
            payload["match_raw_client"] = bool(payload.get("match", False))
            payload["match"] = normalized_match

            save_reason = ""
            save_path: Optional[str] = None

            if self.mode == "collect_retrain":
                roi_b64 = payload.get("roi_jpg_b64") or ""
                confidence = float(payload.get("confidence", 0.0))

                should_save = False
                if roi_b64:
                    if not normalized_match:
                        should_save = True
                        save_reason = "wrong"
                    elif confidence < LOW_CONF_THRESHOLD:
                        should_save = True
                        save_reason = "low_conf"

                if should_save and self._can_save_locked(expected):
                    try:
                        save_path = self._save_roi_locked(
                            expected=expected,
                            predicted=predicted,
                            confidence=confidence,
                            trial_id=str(payload.get("trial_id", self.current_trial.trial_id)),
                            source_name=current_item.name,
                            roi_b64=roi_b64,
                            bucket="hard_examples" if save_reason == "wrong" else "low_confidence",
                        )
                        payload["saved_roi_path"] = save_path
                        payload["save_reason"] = save_reason
                    except Exception as e:
                        payload["save_error"] = str(e)

            with self.results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

            self.last_post_summary = (
                f"trial={payload.get('trial_id','')} expected={expected} "
                f"pred={predicted} canonical=({expected_canonical},{predicted_canonical}) "
                f"match={normalized_match} save={save_reason or 'none'}"
            )

            if AUTO_ADVANCE:
                self.current_index = (self.current_index + 1) % len(self.items)
                self.current_trial = self._new_trial_locked()

            return True, save_path, save_reason

    def _disk_usage_mb_locked(self) -> float:
        total = 0
        for root, _, files in os.walk(self.run_dir):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    pass
        return total / (1024.0 * 1024.0)

    def _can_save_locked(self, label: str) -> bool:
        if self.total_saved >= MAX_SAVED_TOTAL:
            return False
        if self.class_saved_counts.get(label, 0) >= MAX_SAVED_PER_CLASS:
            return False
        if self._disk_usage_mb_locked() >= float(DISK_CAP_MB):
            return False
        return True

    def _save_roi_locked(
        self,
        expected: str,
        predicted: str,
        confidence: float,
        trial_id: str,
        source_name: str,
        roi_b64: str,
        bucket: str,
    ) -> str:
        data = base64.b64decode(roi_b64.encode("ascii"))
        label_dir = self.run_dir / bucket / safe_name(expected)
        label_dir.mkdir(parents=True, exist_ok=True)
        src_stem = Path(source_name).stem
        fname = (
            f"{safe_name(trial_id)}"
            f"__src_{safe_name(src_stem)}"
            f"__exp_{safe_name(expected)}"
            f"__pred_{safe_name(predicted)}"
            f"__{confidence:.4f}.jpg"
        )
        out_path = label_dir / fname
        out_path.write_bytes(data)
        self.total_saved += 1
        self.class_saved_counts[expected] = self.class_saved_counts.get(expected, 0) + 1
        return str(out_path)

    def status_payload(self) -> dict:
        with self.lock:
            return {
                "ok": True,
                "session": self.session,
                "mode": self.mode,
                "current": self.current_trial_payload(),
                "run_dir": str(self.run_dir),
                "workspace_dir": str(self.workspace_dir),
                "results_path": str(self.results_path),
                "saved_total": self.total_saved,
                "class_saved_counts": self.class_saved_counts,
                "last_post_summary": self.last_post_summary,
            }


class App:
    def __init__(self, args: argparse.Namespace) -> None:
        self.host = "0.0.0.0"
        self.port = args.port
        self.state = SessionState(
            dataset_root=Path(args.dataset_root),
            output_root=Path(args.output_root),
            mode=args.mode,
        )

    def make_handler(self):
        app = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "vision_app_controller/5.0"

            def log_message(self, fmt: str, *args) -> None:
                return

            def _send_bytes(self, code: int, content_type: str, data: bytes) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def _send_json(self, obj: dict, code: int = 200) -> None:
                data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self._send_bytes(code, "application/json; charset=utf-8", data)

            def _read_json(self) -> dict:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                return json.loads(raw.decode("utf-8"))

            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path in ("/", "/display"):
                    self._send_bytes(200, "text/html; charset=utf-8", display_html())
                    return
                if path == "/calibrate_tag":
                    self._send_bytes(200, "text/html; charset=utf-8", tag_html())
                    return
                if path == "/img":
                    self._send_bytes(200, "image/png", app.state.current_stimulus_png())
                    return
                if path == "/tag":
                    self._send_bytes(200, "image/png", build_tag_stimulus())
                    return
                if path == "/health":
                    self._send_json({"ok": True})
                    return
                if path == "/api/current":
                    self._send_json(app.state.current_payload())
                    return
                if path == "/api/next":
                    self._send_json(app.state.advance())
                    return
                if path == "/api/status":
                    self._send_json(app.state.status_payload())
                    return
                self._send_json({"ok": False, "error": "not found"}, code=404)

            def do_POST(self) -> None:
                path = urlparse(self.path).path
                if path != "/api/result":
                    self._send_json({"ok": False, "error": "not found"}, code=404)
                    return
                try:
                    payload = self._read_json()
                except Exception as e:
                    self._send_json({"ok": False, "error": f"bad json: {e}"}, code=400)
                    return
                ok, save_path, save_reason = app.state.append_result(payload)
                self._send_json(
                    {
                        "ok": ok,
                        "saved_roi_path": save_path,
                        "save_reason": save_reason,
                        "next": app.state.current_payload(),
                    }
                )

        return Handler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Minimal laptop server: serve demo stimulus, serve AprilTag stimulus, collect Pi results."
    )
    p.add_argument("--dataset-root", required=True, help="Read-only source image root: one class folder per label")
    p.add_argument("--output-root", required=True, help="Writable root. Creates sessions/ and workspaces/ here")
    p.add_argument("--mode", default="demo", choices=["demo", "collect_retrain"], help="demo=log only, collect_retrain=save hard examples")
    p.add_argument("--port", type=int, default=8787, help="HTTP port")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    app = App(args)
    server = ThreadingHTTPServer((app.host, app.port), app.make_handler())

    print(f"controller ready: http://{app.host}:{app.port}/")
    print(f"iPad demo page:   http://<laptop-ip>:{app.port}/display")
    print(f"iPad tag page:    http://<laptop-ip>:{app.port}/calibrate_tag")
    print(f"status api:       http://<laptop-ip>:{app.port}/api/status")
    print(f"session:          {app.state.session}")
    print(f"mode:             {app.state.mode}")
    print(f"dataset items:    {len(app.state.items)}")
    print(f"run dir:          {app.state.run_dir}")
    print(f"workspace dir:    {app.state.workspace_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
