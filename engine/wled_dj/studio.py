"""The studio: a local web page (HTTP + WebSocket) for watching and controlling the engine.

Only 127.0.0.1/localhost Host headers are served, and the WebSocket needs a matching Origin and the
per-run token embedded in the page, so other websites cannot drive the lights."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
from pathlib import Path
import re
import secrets
import threading

from aiohttp import WSMsgType, web

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "studio"
MEDIA_DIR = ROOT / "bench" / "out" / "looks"          # where bench/render_look.py puts its videos and looks.json
MEDIA = re.compile(r"^[\w.-]+\.(mp4|json)$")


STATUS_EVERY_S = .5
FRAME_EVERY_S = 1 / 60          # the preview: as often as a screen redraws


def _local_port(request: web.Request) -> int:
    sockname = request.transport.get_extra_info("sockname") if request.transport else None
    return sockname[1] if sockname else 0


def _host_ok(request: web.Request) -> bool:
    port = _local_port(request)
    return request.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")


def _origin_ok(request: web.Request) -> bool:
    port = _local_port(request)
    return request.headers.get("Origin") in (f"http://127.0.0.1:{port}", f"http://localhost:{port}")


def summary(status: dict) -> str:
    capture = status["capture"]
    if capture["state"] != "running":
        head = {"no-audio": "No audio", "restarting": "Capture restarting", "starting": "Starting"}.get(
            capture["state"], capture["state"].capitalize())
    elif status["output"].get("handback"):
        head = "Idle"
    else:
        head = "Running"
    parts = [head]
    if capture.get("output"):
        parts.append(capture["output"]["name"])
    controller = status["controller"]
    parts.append({"online": f"{controller.get('name') or 'WLED'} online", "offline": "lights offline",
                  "busy": "lights busy", "choose": "choose your lights",
                  "searching": "looking for lights"}.get(controller["state"], controller["state"]))
    return " · ".join(parts)


def _number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected a number")
    return float(value)


def handle_command(engine, message: dict) -> None:
    kind = message.get("type")
    if kind == "look":
        if not isinstance(message.get("id"), str):
            raise ValueError("Pick a look")
        engine.set_look(message["id"])
    elif kind == "master":
        engine.set_master(_number(message.get("value")))
    elif kind == "blackout":
        if not isinstance(message.get("on"), bool):
            raise ValueError("Blackout needs on: true or false")
        engine.set_blackout(message["on"])
    elif kind == "trim":
        engine.set_trim(_number(message.get("ms")))
    elif kind == "stems":
        if not isinstance(message.get("on"), bool):
            raise ValueError("Layer separation needs on: true or false")
        engine.set_stems(message["on"])
    elif kind == "flash_limit":
        if isinstance(message.get("value"), bool) or not isinstance(message.get("value"), int):
            raise ValueError("The flash limit is 0 (off) to 3 a second")
        engine.set_flash_limit(message["value"])
    elif kind == "controller":
        if not isinstance(message.get("host"), str) or not message["host"].strip():
            raise ValueError("Enter a controller address")
        engine.select_controller(message["host"].strip())
    elif kind == "quit":
        engine.quit()
    else:
        raise ValueError(f"Unknown message: {kind}")


def create_app(engine, token: str, static_dir: Path = STATIC, port: int = 0, media_dir: Path = MEDIA_DIR) -> web.Application:

    def page(name: str):
        async def handler(request: web.Request) -> web.Response:
            if not _host_ok(request):
                return web.Response(status=403, text="Forbidden")
            text = (static_dir / name).read_text().replace("__TOKEN__", token)
            return web.Response(text=text, content_type="text/html", headers={"Cache-Control": "no-store"})
        return handler

    async def media(request: web.Request) -> web.StreamResponse:
        name = request.match_info["name"]
        path = media_dir / name
        if not _host_ok(request) or not MEDIA.match(name) or not path.is_file():
            return web.Response(status=404, text="Not found")
        return web.FileResponse(path, headers={"Cache-Control": "no-store"})

    async def health(request: web.Request) -> web.Response:
        if not _host_ok(request):
            return web.Response(status=403, text="Forbidden")
        return web.json_response({"ok": True, "summary": summary(engine.status())})

    async def websocket(request: web.Request) -> web.StreamResponse:
        if not (_host_ok(request) and _origin_ok(request)) or request.query.get("token") != token:
            return web.Response(status=403, text="Forbidden")
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(request)

        async def push() -> None:
            next_status = 0.0
            loop = asyncio.get_running_loop()
            next_frame = loop.time()
            while not ws.closed:
                now = loop.time()
                if now >= next_status:
                    await ws.send_json({"type": "status", **engine.status()})
                    next_status = now + STATUS_EVERY_S
                preview = engine.preview()
                if preview is not None:
                    await ws.send_json({"type": "frame", "pixels": base64.b64encode(preview["pixels"]).decode(),
                                        "meters": preview["meters"], "beat": preview["beat"],
                                        "layers": preview.get("layers", {})})
                next_frame = max(next_frame + FRAME_EVERY_S, loop.time() - FRAME_EVERY_S)      # on time, without a burst after a stall
                await asyncio.sleep(max(0.0, next_frame - loop.time()))

        pusher = asyncio.create_task(push())
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = msg.json()
                    if not isinstance(data, dict):
                        raise ValueError("Expected an object")
                    handle_command(engine, data)
                except ValueError as error:
                    await ws.send_json({"type": "error", "message": str(error)})
        finally:
            pusher.cancel()
        return ws

    app = web.Application()
    app.router.add_get("/", page("index.html"))
    app.router.add_get("/looks", page("looks.html"))
    app.router.add_get("/looks/media/{name}", media)
    app.router.add_get("/api/health", health)
    app.router.add_get("/ws", websocket)
    app.router.add_static("/static", static_dir)
    return app


def serve(engine, port: int, static_dir: Path = STATIC) -> str:
    """Run the studio on a daemon thread; return its URL. Falls back to a free port if `port` is taken."""
    token = secrets.token_urlsafe(24)
    ready = threading.Event()
    result: dict = {}

    def run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(create_app(engine, token, static_dir), access_log=None)
        loop.run_until_complete(runner.setup())
        for candidate in (port, 0):
            try:
                site = web.TCPSite(runner, "127.0.0.1", candidate)
                loop.run_until_complete(site.start())
                result["port"] = site._server.sockets[0].getsockname()[1]
                break
            except OSError as error:
                log.warning("Studio port %s unavailable (%s)", candidate, error)
        ready.set()
        loop.run_forever()

    threading.Thread(target=run, name="studio", daemon=True).start()
    ready.wait(10)
    if "port" not in result:
        raise RuntimeError("Studio could not start")
    return f"http://127.0.0.1:{result['port']}/"
