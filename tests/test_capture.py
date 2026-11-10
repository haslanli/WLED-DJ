import io
import json
import os
from pathlib import Path
import struct
import sys
import threading
import time

import numpy as np
import pytest

from wled_dj.capture import AudioBlock, CaptureClient, ProtocolError, read_messages

FAKE = str(Path(__file__).parent / "fakes" / "fake_capture.py")


def frame(kind, payload):
    return struct.pack("<cI", kind, len(payload)) + payload


def fake_cmd(scenario, *extra):
    return [sys.executable, FAKE, "--scenario", scenario, *extra]


class Recorder:
    def __init__(self):
        self.audio, self.outputs, self.statuses = [], [], []
        self.lock = threading.Lock()

    def on_audio(self, block):
        with self.lock:
            self.audio.append((time.monotonic(), block))

    def on_output(self, info):
        with self.lock:
            self.outputs.append(info)

    def on_status(self, status):
        with self.lock:
            self.statuses.append((time.monotonic(), status))

    def wait(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if predicate(self):
                    return True
            time.sleep(.02)
        return False


def client(scenario, rec):
    return CaptureClient(fake_cmd(scenario), rec.on_audio, rec.on_output, rec.on_status)


def test_read_messages_parses_audio_and_json():
    samples = np.arange(960, dtype="<f4") / 960
    stream = (frame(b"J", b'{"event": "ready"}')
              + frame(b"A", struct.pack("<dI", 12.5, 3) + samples.tobytes())
              + frame(b"J", json.dumps({"event": "output", "name": "X", "uid": "x", "transport": "bltn",
                                        "latency_ms": 22.1}).encode()))
    items = list(read_messages(io.BytesIO(stream)))
    assert items[0] == {"event": "ready"} and items[1].samples.shape == (480, 2)
    assert items[1].host_time == 12.5 and items[1].drops == 3 and items[2]["latency_ms"] == 22.1
    assert isinstance(items[1], AudioBlock) and items[1].samples[1, 0] == np.float32(2 / 960)


def test_read_messages_rejects_oversize():
    with pytest.raises(ProtocolError):
        list(read_messages(io.BytesIO(struct.pack("<cI", b"A", 2 * 1024 * 1024))))


def test_truncated_stream_raises():
    with pytest.raises(ProtocolError, match="truncated"):
        list(read_messages(io.BytesIO(frame(b"A", struct.pack("<dI", 1.0, 0) + b"\0" * 16)[:-5])))


def test_client_restarts_after_helper_exit():
    rec = Recorder()
    c = client("exit-after-3-blocks", rec)
    c.start()
    try:
        assert rec.wait(lambda r: c.launches >= 2, 3.0)
        assert any(s.state == "restarting" for _, s in rec.statuses)
        seen = len(rec.audio)
        assert rec.wait(lambda r: len(r.audio) > seen, 3.0)
    finally:
        c.stop()


def test_helper_error_message_surfaces():
    rec = Recorder()
    c = client("error-exit", rec)
    c.start()
    try:
        assert rec.wait(lambda r: any(s.state == "restarting" and "1852797029" in s.message
                                      for _, s in r.statuses), 3.0)
    finally:
        c.stop()


def test_no_audio_status():
    rec = Recorder()
    c = client("no-audio", rec)
    c.start()
    try:
        assert rec.wait(lambda r: any(s.state == "no-audio" for _, s in r.statuses), 6.0)
        message = next(s.message for _, s in rec.statuses if s.state == "no-audio")
        assert "Privacy & Security" in message and "Screen & System Audio Recording" in message
    finally:
        c.stop()


def test_running_status_and_output_reported():
    rec = Recorder()
    c = client("music", rec)
    c.start()
    try:
        assert rec.wait(lambda r: any(s.state == "running" for _, s in r.statuses) and r.outputs, 3.0)
        assert rec.outputs[0].latency_ms == 22.1 and rec.outputs[0].uid == "fake-speakers"
        _, last = rec.statuses[-1]
        assert last.output is not None and last.output.name == "Fake Speakers"
    finally:
        c.stop()


def test_stop_terminates_helper():
    rec = Recorder()
    c = client("music", rec)
    c.start()
    assert rec.wait(lambda r: len(r.audio) > 2, 3.0)
    pid = c.pid
    c.stop()
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(.05)
    else:
        pytest.fail("helper still alive")
    assert rec.statuses[-1][1].state == "stopped"
