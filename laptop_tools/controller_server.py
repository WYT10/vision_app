#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import random
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


@dataclass
class TrialItem:
    label: str
    image_path: Path


class ControllerState:
    def __init__(self,
                 dataset_root: Path,
                 output_root: Path,
                 workspace_root: Path | None,
                 mode: str,
                 auto_advance: bool,
                 seed: int,
                 low_conf_threshold: float,
                 max_saved_per_class: int,
                 max_saved_total: int,
                 disk_cap_mb: int,
                 retrain_threshold: int,
                 auto_run_retrain_cmd: str,
                 save_wrong_only: bool) -> None:
        self.dataset_root = dataset_root.resolve()
        self.output_root = output_root.resolve()
        self.sessions_root = self.output_root / 'sessions'
        default_workspace_root = self.output_root / 'workspaces'
        self.workspaces_root = (workspace_root.resolve() if workspace_root else default_workspace_root.resolve())
        self.mode = mode
        self.auto_advance = auto_advance
        self.low_conf_threshold = low_conf_threshold
        self.max_saved_per_class = max_saved_per_class
        self.max_saved_total = max_saved_total
        self.max_saved_bytes = max(0, disk_cap_mb) * 1024 * 1024
        self.retrain_threshold = retrain_threshold
        self.auto_run_retrain_cmd = auto_run_retrain_cmd
        self.save_wrong_only = save_wrong_only
        self.rng = random.Random(seed)
        self.items: List[TrialItem] = self._scan_items(dataset_root)
        if not self.items:
            raise RuntimeError(f'No images found under {dataset_root}')
        self.session = time.strftime('session_%Y%m%d_%H%M%S')
        self.version = 0
        self.index = 0
        self.trial_id = self._make_trial_id()
        self.lock = threading.Lock()
        self.current_result: Optional[dict] = None
        self.run_dir = self.sessions_root / self.session
        self.workspace_dir = self.workspaces_root / self.session
        self.results_path = self.run_dir / 'results.jsonl'
        self.hard_dir = self.run_dir / 'hard_examples'
        self.lowconf_dir = self.run_dir / 'low_confidence'
        self.correct_dir = self.run_dir / 'correct'
        self.rejected_dir = self.run_dir / 'rejected'
        for d in (self.sessions_root, self.workspaces_root, self.run_dir, self.workspace_dir,
                  self.hard_dir, self.lowconf_dir, self.correct_dir, self.rejected_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.saved_total = 0
        self.saved_bytes = 0
        self.saved_per_class: Dict[str, int] = {}
        self.retrain_requested = False
        self.retrain_running = False
        self.last_retrain_status = 'idle'
        self.session_info_path = self.run_dir / 'session.json'
        self._write_session_info()

    def _scan_items(self, root: Path) -> List[TrialItem]:
        items: List[TrialItem] = []
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir() or class_dir.name.startswith('.'):
                continue
            for p in sorted(class_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith('.'):
                    items.append(TrialItem(label=class_dir.name, image_path=p))
        return items

    def _make_trial_id(self) -> str:
        return f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"

    def _write_session_info(self) -> None:
        info = {
            'session': self.session,
            'mode': self.mode,
            'dataset_root': str(self.dataset_root),
            'output_root': str(self.output_root),
            'sessions_root': str(self.sessions_root),
            'workspaces_root': str(self.workspaces_root),
            'run_dir': str(self.run_dir),
            'workspace_dir': str(self.workspace_dir),
            'low_conf_threshold': self.low_conf_threshold,
            'max_saved_per_class': self.max_saved_per_class,
            'max_saved_total': self.max_saved_total,
            'disk_cap_mb': self.max_saved_bytes / (1024 * 1024) if self.max_saved_bytes else 0,
            'retrain_threshold': self.retrain_threshold,
            'auto_run_retrain_cmd': self.auto_run_retrain_cmd,
            'save_wrong_only': self.save_wrong_only,
            'items': len(self.items),
        }
        self.session_info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')

    def current_item(self) -> TrialItem:
        return self.items[self.index]

    def current_payload(self) -> dict:
        item = self.current_item()
        return {
            'ok': True,
            'active': True,
            'session': self.session,
            'mode': self.mode,
            'trial_id': self.trial_id,
            'label': item.label,
            'image_name': item.image_path.name,
            'image_url': '/img',
            'version': self.version,
            'retrain_requested': self.retrain_requested,
            'retrain_running': self.retrain_running,
        }

    def advance(self, mode: str = 'next') -> dict:
        with self.lock:
            if mode == 'random':
                self.index = self.rng.randrange(len(self.items))
            else:
                self.index = (self.index + 1) % len(self.items)
            self.version += 1
            self.trial_id = self._make_trial_id()
            return self.current_payload()

    def _save_image_if_allowed(self, payload: dict, bucket_dir: Path) -> str:
        expected = payload.get('expected_label') or 'unknown'
        if self.max_saved_total > 0 and self.saved_total >= self.max_saved_total:
            payload['save_skip_reason'] = 'max_saved_total'
            return ''
        if self.max_saved_per_class > 0 and self.saved_per_class.get(expected, 0) >= self.max_saved_per_class:
            payload['save_skip_reason'] = 'max_saved_per_class'
            return ''
        roi_jpg_b64 = payload.get('roi_jpg_b64') or ''
        if not roi_jpg_b64:
            payload['save_skip_reason'] = 'no_roi'
            return ''
        img_bytes = base64.b64decode(roi_jpg_b64)
        if self.max_saved_bytes > 0 and self.saved_bytes + len(img_bytes) > self.max_saved_bytes:
            payload['save_skip_reason'] = 'disk_cap'
            return ''
        bucket_dir.mkdir(parents=True, exist_ok=True)
        confidence = float(payload.get('confidence', 0.0) or 0.0)
        fname = f"{payload.get('trial_id','trial')}__pred_{payload.get('predicted_label','none')}__{confidence:.3f}.jpg"
        out_path = bucket_dir / fname
        out_path.write_bytes(img_bytes)
        self.saved_total += 1
        self.saved_bytes += len(img_bytes)
        self.saved_per_class[expected] = self.saved_per_class.get(expected, 0) + 1
        return str(out_path)

    def _maybe_launch_retrain(self) -> None:
        if self.mode != 'collect_retrain':
            return
        if self.retrain_threshold <= 0 or self.saved_total < self.retrain_threshold:
            return
        if self.retrain_requested or self.retrain_running:
            return
        self.retrain_requested = True
        if not self.auto_run_retrain_cmd:
            self.last_retrain_status = 'threshold_reached_waiting_manual'
            return

        def worker() -> None:
            self.retrain_running = True
            self.last_retrain_status = 'running'
            cmd = self.auto_run_retrain_cmd.format(
                run_dir=str(self.run_dir),
                workspace_dir=str(self.workspace_dir),
                session=self.session,
                output_root=str(self.output_root),
                sessions_root=str(self.sessions_root),
                workspaces_root=str(self.workspaces_root),
            )
            try:
                rc = subprocess.call(cmd, shell=True)
                self.last_retrain_status = f'finished_rc_{rc}'
            except Exception as e:
                self.last_retrain_status = f'error:{e}'
            finally:
                self.retrain_running = False

        threading.Thread(target=worker, daemon=True).start()

    def save_result(self, payload: dict) -> dict:
        with self.lock:
            payload = dict(payload)
            payload['server_time'] = time.time()
            payload['session'] = payload.get('session') or self.session
            payload['server_mode'] = self.mode
            expected = payload.get('expected_label') or 'unknown'
            match = bool(payload.get('match', False))
            confidence = float(payload.get('confidence', 0.0) or 0.0)
            saved_path = ''
            if self.mode == 'collect_retrain':
                should_save = (not match) or (confidence < self.low_conf_threshold and not self.save_wrong_only)
                if should_save:
                    bucket_dir = (self.lowconf_dir / expected) if match else (self.hard_dir / expected)
                    saved_path = self._save_image_if_allowed(payload, bucket_dir)
            payload['saved_roi_path'] = saved_path
            with self.results_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')
            self.current_result = payload
            self._maybe_launch_retrain()
            if self.auto_advance:
                self.index = (self.index + 1) % len(self.items)
                self.version += 1
                self.trial_id = self._make_trial_id()
            return {'ok': True, 'saved_roi_path': saved_path, 'next': self.current_payload(), 'saved_total': self.saved_total, 'retrain_requested': self.retrain_requested, 'retrain_running': self.retrain_running}

    def status(self) -> dict:
        return {
            'ok': True,
            'session': self.session,
            'mode': self.mode,
            'items': len(self.items),
            'saved_total': self.saved_total,
            'saved_per_class': self.saved_per_class,
            'saved_bytes': self.saved_bytes,
            'retrain_requested': self.retrain_requested,
            'retrain_running': self.retrain_running,
            'last_retrain_status': self.last_retrain_status,
            'output_root': str(self.output_root),
            'run_dir': str(self.run_dir),
            'workspace_dir': str(self.workspace_dir),
            'current': self.current_payload(),
        }


def make_handler(state: ControllerState):
    class Handler(BaseHTTPRequestHandler):
        def _send_bytes(self, code: int, data: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, obj: dict, code: int = 200) -> None:
            self._send_bytes(code, json.dumps(obj, ensure_ascii=False).encode('utf-8'), 'application/json')

        def _read_json(self) -> dict:
            n = int(self.headers.get('Content-Length', '0'))
            raw = self.rfile.read(n) if n > 0 else b'{}'
            return json.loads(raw.decode('utf-8'))

        def log_message(self, format: str, *args):
            return

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            if path == '/api/current':
                with state.lock:
                    self._send_json(state.current_payload())
                return
            if path == '/api/status':
                with state.lock:
                    self._send_json(state.status())
                return
            if path == '/api/next':
                mode = parse_qs(parsed.query).get('mode', ['next'])[0]
                self._send_json(state.advance(mode=mode))
                return
            if path == '/img':
                with state.lock:
                    img_path = state.current_item().image_path
                ctype = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.bmp': 'image/bmp', '.webp': 'image/webp'}.get(img_path.suffix.lower(), 'application/octet-stream')
                self._send_bytes(200, img_path.read_bytes(), ctype)
                return
            if path == '/api/last_result':
                with state.lock:
                    self._send_json(state.current_result or {'ok': False})
                return
            if path == '/display':
                html = f"""<!doctype html>
<html><head><meta name='viewport' content='width=device-width, initial-scale=1' />
<style>
html,body{{margin:0;background:#111;color:#fff;font-family:sans-serif;height:100%;}}
#wrap{{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;gap:10px;}}
img{{max-width:98vw;max-height:88vh;object-fit:contain;}}
#meta{{font-size:18px;text-align:center;padding:0 12px;}}
</style></head><body><div id='wrap'><img id='img' src='/img'><div id='meta'>loading…</div></div>
<script>
let lastTrial='';
async function tick(){{
  const r = await fetch('/api/current', {{cache:'no-store'}});
  const j = await r.json();
  document.getElementById('meta').textContent = `${{j.mode}} | ${{j.label}} | ${{j.image_name}} | ${{j.trial_id}}`;
  if (j.trial_id !== lastTrial) {{
    document.getElementById('img').src = '/img?t=' + encodeURIComponent(j.trial_id);
    lastTrial = j.trial_id;
  }}
}}
setInterval(tick, 300); tick();
</script></body></html>"""
                self._send_bytes(200, html.encode('utf-8'), 'text/html; charset=utf-8')
                return
            if path == '/':
                status = state.status()
                html = f"""<!doctype html><html><body>
<h2>Stimulus controller</h2>
<p>session: {state.session}</p>
<p>mode: {state.mode}</p>
<p>run_dir: {state.run_dir}</p>
<ul>
<li><a href='/display'>/display</a> — open this on iPad</li>
<li><a href='/api/current'>/api/current</a></li>
<li><a href='/api/status'>/api/status</a></li>
<li><a href='/api/next'>/api/next</a></li>
<li><a href='/api/next?mode=random'>/api/next?mode=random</a></li>
<li><a href='/api/last_result'>/api/last_result</a></li>
</ul>
<pre>{json.dumps(status, indent=2, ensure_ascii=False)}</pre>
</body></html>"""
                self._send_bytes(200, html.encode('utf-8'), 'text/html; charset=utf-8')
                return
            self._send_json({'ok': False, 'error': 'not found'}, code=404)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path == '/api/result':
                payload = self._read_json()
                self._send_json(state.save_result(payload), code=201)
                return
            if parsed.path == '/api/next':
                payload = self._read_json() if self.headers.get('Content-Length') else {}
                mode = payload.get('mode', 'next')
                self._send_json(state.advance(mode=mode))
                return
            if parsed.path == '/api/retrain':
                with state.lock:
                    state.retrain_requested = True
                    state._maybe_launch_retrain()
                    self._send_json({'ok': True, 'retrain_requested': state.retrain_requested, 'retrain_running': state.retrain_running, 'last_retrain_status': state.last_retrain_status})
                return
            self._send_json({'ok': False, 'error': 'not found'}, code=404)

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description='Laptop controller server for iPad display + Pi result collection.')
    ap.add_argument('--dataset-root', required=True, help='Folder with class subfolders of images displayed on the iPad')
    ap.add_argument('--output-root', default='runs/controller', help='Writable root. Server creates sessions/ and workspaces/ under this folder')
    ap.add_argument('--workspace-root', default='', help='Optional override for training workspaces root; default is <output-root>/workspaces')
    ap.add_argument('--mode', choices=['demo', 'collect_retrain'], default='demo')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8787)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-auto-advance', action='store_true')
    ap.add_argument('--low-conf-threshold', type=float, default=0.65)
    ap.add_argument('--save-wrong-only', action='store_true', help='In collect_retrain, save only wrong predictions and skip low-confidence correct samples')
    ap.add_argument('--max-saved-per-class', type=int, default=200)
    ap.add_argument('--max-saved-total', type=int, default=3000)
    ap.add_argument('--disk-cap-mb', type=int, default=2048)
    ap.add_argument('--retrain-threshold', type=int, default=300, help='When saved_total reaches this, mark retrain_requested (and optionally launch a command)')
    ap.add_argument('--auto-run-retrain-cmd', default='', help='Optional shell command launched once when retrain threshold is reached; may use {run_dir} and {session}')
    args = ap.parse_args()

    workspace_root = Path(args.workspace_root).resolve() if args.workspace_root else None
    state = ControllerState(
        dataset_root=Path(args.dataset_root).resolve(),
        output_root=Path(args.output_root).resolve(),
        workspace_root=workspace_root,
        mode=args.mode,
        auto_advance=not args.no_auto_advance,
        seed=args.seed,
        low_conf_threshold=args.low_conf_threshold,
        max_saved_per_class=args.max_saved_per_class,
        max_saved_total=args.max_saved_total,
        disk_cap_mb=args.disk_cap_mb,
        retrain_threshold=args.retrain_threshold,
        auto_run_retrain_cmd=args.auto_run_retrain_cmd,
        save_wrong_only=args.save_wrong_only,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f'controller ready: http://{args.host}:{args.port}/')
    print(f'iPad open:        http://<laptop-ip>:{args.port}/display')
    print(f'session:          {state.session}')
    print(f'mode:             {state.mode}')
    print(f'dataset items:    {len(state.items)}')
    print(f'output root:      {state.output_root}')
    print(f'run dir:          {state.run_dir}')
    print(f'workspace dir:    {state.workspace_dir}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
