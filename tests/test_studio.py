import asyncio
import json
import base64
from pathlib import Path

import pytest

from wled_dj.studio import create_app

STATIC = Path(__file__).resolve().parents[1] / "studio"
TOKEN = "test-token"


class FakeEngine:
    def __init__(self):
        self.calls = []

    def status(self):
        return {"capture": {"state": "running", "message": "", "output": {"name": "Speakers", "uid": "s",
                                                                          "latency_ms": 22.1}},
                "sync": {"delay_ms": 0.0, "lead_ms": 2.9, "trim_ms": 0.0, "processing_ms": 15.0},
                "controller": {"state": "online", "host": "192.168.1.50", "name": "WLED", "pixels": 78,
                               "buses": [{"start": 0, "length": 78, "reversed": False}], "fps": 77,
                               "candidates": []},
                "output": {"sent": 10, "skipped": 0, "errors": 0, "latency_ms": 15.0, "interval_p99_ms": 13.5,
                           "handback": False, "audio_dropped": 0},
                "look": "sync-check", "looks": [{"id": "sync-check", "name": "Sync check"}], "master": 1.0, "blackout": False,
                "music": {"bpm": 128.0, "confidence": .8, "silent": False},
                "layers": {"enabled": True, "state": "on", "source": "stems", "late_share": 0.0, "message": ""},
                "structure": {"section": "breakdown", "energy": .62, "tension": .1, "colour": 74.0, "lead": .31},
                "flash_limit": 3}

    def preview(self):
        return {"pixels": bytes(range(234)), "meters": {"kick": .5, "snare": 0, "hats": 0, "bass": 0, "vocal": 0},
                "beat": .25, "layers": {name: {"level": .5, "present": name == "drums"}
                                        for name in ("drums", "bass", "vocals", "other")}}

    def set_stems(self, on):
        self.calls.append(("stems", on))

    def set_flash_limit(self, value):
        self.calls.append(("flash_limit", value))

    def set_look(self, look_id):
        if look_id != "sync-check":
            raise ValueError("Unknown look")
        self.calls.append(("look", look_id))

    def set_master(self, value):
        self.calls.append(("master", value))

    def set_blackout(self, on):
        self.calls.append(("blackout", on))

    def set_trim(self, ms):
        self.calls.append(("trim", ms))

    def select_controller(self, host, name=""):
        self.calls.append(("controller", host))

    def quit(self):
        self.calls.append(("quit",))


@pytest.fixture
async def studio(aiohttp_client):
    engine = FakeEngine()
    client = await aiohttp_client(create_app(engine, TOKEN, STATIC, port=0))
    port = client.server.port
    return engine, client, port


async def test_index_requires_local_host(studio):
    engine, client, port = studio
    bad = await client.get("/", headers={"Host": "evil.com"})
    assert bad.status == 403
    good = await client.get("/", headers={"Host": f"127.0.0.1:{port}"})
    assert good.status == 200 and TOKEN in await good.text()
    assert "__TOKEN__" not in await good.text()


async def test_ws_rejects_bad_origin_or_token(studio):
    engine, client, port = studio
    headers = {"Host": f"127.0.0.1:{port}", "Origin": "http://evil.com"}
    bad_origin = await client.get(f"/ws?token={TOKEN}", headers=headers)
    assert bad_origin.status == 403
    headers["Origin"] = f"http://127.0.0.1:{port}"
    bad_token = await client.get("/ws?token=nope", headers=headers)
    assert bad_token.status == 403


async def connect(client, port):
    return await client.ws_connect(f"/ws?token={TOKEN}", headers={"Host": f"127.0.0.1:{port}",
                                                                  "Origin": f"http://127.0.0.1:{port}"})


async def test_ws_streams_status_and_frames(studio):
    engine, client, port = studio
    ws = await connect(client, port)
    kinds = []
    pixels = None
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 1.0
    while loop.time() < deadline:
        msg = await ws.receive_json(timeout=1.0)
        kinds.append(msg["type"])
        if msg["type"] == "frame":
            pixels = base64.b64decode(msg["pixels"])
    await ws.close()
    assert kinds.count("status") >= 1 and kinds.count("frame") >= 5
    assert len(pixels) == 234


async def test_ws_commands_call_engine(studio):
    engine, client, port = studio
    ws = await connect(client, port)
    for message in ({"type": "look", "id": "sync-check"}, {"type": "master", "value": .4},
                    {"type": "blackout", "on": True}, {"type": "trim", "ms": -12},
                    {"type": "controller", "host": "10.0.0.9"}, {"type": "quit"}):
        await ws.send_json(message)
    await ws.send_json({"type": "master", "value": "x"})
    await ws.send_json({"type": "look", "id": "nope"})
    errors = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 1.0
    while loop.time() < deadline and len(errors) < 2:
        msg = await ws.receive_json(timeout=1.0)
        if msg["type"] == "error":
            errors.append(msg["message"])
    await ws.close()
    assert engine.calls == [("look", "sync-check"), ("master", .4), ("blackout", True), ("trim", -12),
                            ("controller", "10.0.0.9"), ("quit",)]
    assert len(errors) == 2


async def test_health(studio):
    engine, client, port = studio
    response = await client.get("/api/health", headers={"Host": f"localhost:{port}"})
    body = await response.json()
    assert response.status == 200 and body["ok"] is True and isinstance(body["summary"], str)
    assert "Speakers" in body["summary"]


@pytest.fixture
async def looks_studio(aiohttp_client, tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    (media / "looks.json").write_text('{"clips": [{"id": "one", "title": "One", "group": "clips"}], "looks": {}}')
    (media / "glow--one.mp4").write_bytes(b"\0" * 16)
    (tmp_path / "secret.txt").write_text("no")
    engine = FakeEngine()
    client = await aiohttp_client(create_app(engine, TOKEN, STATIC, port=0, media_dir=media))
    return client, client.server.port, tmp_path


async def test_looks_media_serves_only_renders(looks_studio):
    client, port, _ = looks_studio
    host = {"Host": f"127.0.0.1:{port}"}
    index = await client.get("/looks/media/looks.json", headers=host)
    assert index.status == 200 and (await index.json())["clips"][0]["id"] == "one"
    assert (await client.get("/looks/media/glow--one.mp4", headers=host)).status == 200
    assert (await client.get("/looks/media/..%2Fsecret.txt", headers=host)).status == 404
    assert (await client.get("/looks/media/notes.txt", headers=host)).status == 404
    assert (await client.get("/looks/media/looks.json", headers={"Host": "evil.com"})).status == 404


async def test_frames_carry_the_layers(studio):
    engine, client, port = studio
    ws = await connect(client, port)
    frame = None
    while frame is None:
        msg = await ws.receive_json(timeout=1.0)
        if msg["type"] == "frame":
            frame = msg
    await ws.close()
    assert frame["layers"] == engine.preview()["layers"]


async def test_stems_switch(studio):
    engine, client, port = studio
    ws = await connect(client, port)
    await ws.send_json({"type": "stems", "on": False})
    await ws.send_json({"type": "stems", "on": "no"})
    error = None
    while error is None:
        msg = await ws.receive_json(timeout=1.0)
        if msg["type"] == "error":
            error = msg["message"]
    await ws.close()
    assert engine.calls == [("stems", False)] and "on" in error


async def test_page_has_the_layer_meters_and_switch(studio):
    engine, client, port = studio
    page = await (await client.get("/", headers={"Host": f"127.0.0.1:{port}"})).text()
    for name in ("drums", "bass", "vocals", "other"):
        assert f'data-layer="{name}"' in page
    assert 'id="layers-note"' in page and 'id="stems"' in page


async def test_page_has_the_structure_line(studio):
    engine, client, port = studio
    page = await (await client.get("/", headers={"Host": f"127.0.0.1:{port}"})).text()
    script = await (await client.get("/static/studio.js", headers={"Host": f"127.0.0.1:{port}"})).text()
    assert 'id="structure-note"' in page and '"structure-note"' in script


async def test_flash_limit_message(studio):
    engine, client, port = studio
    ws = await connect(client, port)
    await ws.send_json({"type": "flash_limit", "value": 2})
    await ws.send_json({"type": "flash_limit", "value": "3"})
    error = None
    while error is None:
        msg = await ws.receive_json(timeout=1.0)
        if msg["type"] == "error":
            error = msg["message"]
    await ws.close()
    assert engine.calls == [("flash_limit", 2)] and "flash limit" in error


async def test_page_has_the_flash_limit(studio):
    engine, client, port = studio
    page = await (await client.get("/", headers={"Host": f"127.0.0.1:{port}"})).text()
    script = await (await client.get("/static/studio.js", headers={"Host": f"127.0.0.1:{port}"})).text()
    assert 'id="flash-limit"' in page and "seizures" in page and '"flash-limit"' in script and "flash_limit" in script


async def test_the_preview_comes_sixty_times_a_second(studio):
    """30 a second made every look seem choppy on the page, whatever the lights were doing."""
    engine, client, port = studio
    ws = await connect(client, port)
    loop = asyncio.get_running_loop()
    await ws.receive_json(timeout=1.0)
    started, frames = loop.time(), 0
    while loop.time() - started < 1.0:
        if (await ws.receive_json(timeout=1.0))["type"] == "frame":
            frames += 1
    await ws.close()
    assert 52 <= frames <= 66, frames


# ---- the looks page ---------------------------------------------------------------------------------------

async def test_looks_page_is_served(looks_studio):
    client, port, _ = looks_studio
    assert (await client.get("/looks", headers={"Host": "evil.com"})).status == 403
    page = await client.get("/looks", headers={"Host": f"127.0.0.1:{port}"})
    text = await page.text()
    assert page.status == 200 and TOKEN in text and 'id="clips"' in text and "/static/looks.js" in text
    script = await client.get("/static/looks.js", headers={"Host": f"127.0.0.1:{port}"})
    assert script.status == 200 and "looks.json" in await script.text()


async def test_the_studio_links_to_the_looks_page(studio):
    engine, client, port = studio
    page = await (await client.get("/", headers={"Host": f"127.0.0.1:{port}"})).text()
    assert 'href="/looks"' in page and "Compare the looks" in page


async def test_the_link_to_the_looks_page_waits_for_an_engine_that_serves_it(studio):
    """The app serves the studio's files from disk as they are, while its engine may be one started before the
    looks page existed: the link stays hidden until the page answers."""
    engine, client, port = studio
    host = {"Host": f"127.0.0.1:{port}"}
    page = await (await client.get("/", headers=host)).text()
    script = await (await client.get("/static/studio.js", headers=host)).text()
    assert 'id="looks-link" href="/looks" hidden' in page
    assert '"looks-link"' in script and 'fetch("/looks"' in script
    assert (await client.head("/looks", headers=host)).status == 200
