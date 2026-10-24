"""Small loopback-only control page for the V10 Viser physics viewer."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, urlsplit


def start_demo_ui(host: str, port: int, viewer_port: int, viewer: Any,
                  command: Any, policy: Any, motions: list[dict]) -> ThreadingHTTPServer:
    html = Path(__file__).with_name("demo_ui.html").read_text(encoding="utf-8")
    viewer_url = f"http://{host}:{viewer_port}/"
    html = html.replace("__VISER_URL__", viewer_url)
    ordered = sorted(range(len(motions)), key=lambda i: (not motions[i].get("featured", False), i))
    searchable = [(item["label"] + " " + item["source_name"]).casefold() for item in motions]

    class Handler(BaseHTTPRequestHandler):
        def _json(self, data: Any, status: int = 200) -> None:
            raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            url = urlsplit(self.path)
            if url.path == "/":
                raw = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif url.path == "/api/catalog":
                params = parse_qs(url.query)
                try:
                    query = params.get("q", [""])[0].strip().casefold()
                    offset = max(0, int(params.get("offset", ["0"])[0]))
                    limit = min(100, max(1, int(params.get("limit", ["60"])[0])))
                except ValueError:
                    self._json({"error": "Invalid catalogue page"}, 400)
                    return
                matches = [i for i in ordered if query in searchable[i]]
                self._json({"total": len(matches), "library_total": len(motions),
                            "offset": offset,
                            "motions": [{"index": i, **motions[i]} for i in matches[offset:offset + limit]]})
            elif url.path == "/api/state":
                motion_index = command._viewer_clip_index or 0
                self._json({"paused": viewer._is_paused,
                            "steps": viewer._step_count,
                            "motion": motion_index,
                            "motion_info": motions[motion_index],
                            "policy": policy.enabled,
                            "goal": command._debug_vis_enabled,
                            "noise": policy.noise_std,
                            "friction": 1.0})
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            if self.path != "/api/action":
                self.send_error(404)
                return
            origin = self.headers.get("Origin")
            if origin and origin not in (f"http://{host}:{port}", f"http://localhost:{port}",
                                         f"http://127.0.0.1:{port}"):
                self.send_error(403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1024:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                action = data["action"]
                if action == "play":
                    viewer.request_toggle_pause()
                elif action == "reset":
                    viewer.request_reset()
                elif action == "step":
                    viewer.request_single_step()
                elif action == "select":
                    index = int(data["index"])
                    if not 0 <= index < len(motions):
                        raise ValueError("Unknown motion")
                    command._viewer_clip_index = index
                    viewer.request_reset()
                elif action == "policy":
                    policy.enabled = bool(data["value"])
                elif action == "goal":
                    command._debug_vis_enabled = bool(data["value"])
                    if hasattr(viewer, "_scene"):
                        if viewer._is_paused:
                            # A paused Viser scene does not reliably requeue
                            # debug meshes. Advance one 20 ms frame so its
                            # regular scene pipeline refreshes the overlay.
                            viewer.request_single_step()
                        viewer._scene.request_update()
                elif action == "noise":
                    policy.noise_std = max(0.0, min(float(data["value"]), 0.3))
                else:
                    raise ValueError("Unknown action")
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, 400)
                return
            self._json({"ok": True})

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    Thread(target=server.serve_forever, name="v10-demo-ui", daemon=True).start()
    return server
