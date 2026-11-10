"""A stand-in WLED controller: JSON API over HTTP and a UDP socket that records DDP packets."""
from __future__ import annotations

import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import time


def default_info(pixels=78):
    return {"name": "Fake WLED", "ver": "16.0.1", "leds": {"count": pixels}, "live": False,
            "ddp_fast_refresh": True, "uptime": 5000}


def default_cfg(pixels=78):
    return {"hw": {"led": {"ins": [{"start": 0, "len": pixels, "rev": False},
                                   {"start": 0, "len": pixels, "rev": False}]}},
            "light": {"gc": {"col": 2.2}}, "if": {"live": {"en": True, "no-gc": False}}}


def _merge(target, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value


class FakeWled:
    def __init__(self, pixels=78):
        self.info = default_info(pixels)
        self.state = {"on": True, "bri": 128, "lor": 0}
        self.cfg = default_cfg(pixels)
        self.posts: list[tuple[str, dict]] = []
        self.auto_live = False      # like real WLED: "live" for 2.5 s after the last DDP packet
        self.packets: list[tuple[float, bytes]] = []
        self.lock = threading.Lock()
        self.http_port = self._free_port(socket.SOCK_STREAM)
        self.ddp_port = self._free_port(socket.SOCK_DGRAM)
        self._http = None
        self._udp = None
        self._running = False

    @staticmethod
    def _free_port(kind):
        with socket.socket(socket.AF_INET, kind) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    @property
    def host(self):
        return "127.0.0.1"

    def start(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                key = self.path.rsplit("/", 1)[-1]
                with fake.lock:
                    data = copy.deepcopy({"info": fake.info, "state": fake.state, "cfg": fake.cfg}.get(key))
                    if key == "info" and fake.auto_live and fake.packets and time.monotonic() - fake.packets[-1][0] < 2.5:
                        data["live"] = True
                if data is None:
                    self.send_error(404)
                else:
                    self._send(data)

            def do_POST(self):
                key = self.path.rsplit("/", 1)[-1]
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                with fake.lock:
                    fake.posts.append((key, body))
                    _merge({"state": fake.state, "cfg": fake.cfg}[key], body)
                self._send({"success": True})

        ThreadingHTTPServer.allow_reuse_address = True
        self._http = ThreadingHTTPServer(("127.0.0.1", self.http_port), Handler)
        self._http.daemon_threads = True
        threading.Thread(target=self._http.serve_forever, daemon=True).start()
        self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._udp.bind(("127.0.0.1", self.ddp_port))
        self._udp.settimeout(.1)
        self._running = True
        threading.Thread(target=self._receive, daemon=True).start()
        return self

    def _receive(self):
        sock = self._udp
        while self._running:
            try:
                data, _ = sock.recvfrom(2048)
            except (socket.timeout, OSError):
                continue
            with self.lock:
                self.packets.append((time.monotonic(), data))

    def stop(self):
        self._running = False
        if self._http is not None:
            self._http.shutdown()
            self._http.server_close()
            self._http = None
        if self._udp is not None:
            self._udp.close()
            self._udp = None

    def state_posts(self):
        with self.lock:
            return [body for key, body in self.posts if key == "state"]

    def packet_count(self):
        with self.lock:
            return len(self.packets)
