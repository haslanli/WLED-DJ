"""Client for the native capture helper: framed stdout protocol, supervision and restarts.

Framing: struct "<cI" (type, payload length) followed by the payload.
  b"A": struct "<dI" (host time of the first sample, cumulative drops) + float32 LE stereo at 48 kHz
  b"J": UTF-8 JSON object with an "event" key: ready | output | error
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import struct
import subprocess
import threading
import time
from typing import BinaryIO, Callable, Iterator, Literal

import numpy as np

log = logging.getLogger(__name__)

HEADER = struct.Struct("<cI")
AUDIO_HEADER = struct.Struct("<dI")
MAX_PAYLOAD = 1024 * 1024
NO_AUDIO_AFTER_S = 5.0
HEALTHY_RESET_S = 30.0
BACKOFF_S = (0.5, 1.0, 2.0, 4.0, 8.0, 10.0)
NO_AUDIO_MESSAGE = ("No audio is arriving. Play some music, and make sure WLED DJ is allowed under "
                    "System Settings → Privacy & Security → Screen & System Audio Recording.")


class ProtocolError(Exception):
    pass


@dataclass(frozen=True)
class AudioBlock:
    samples: np.ndarray
    host_time: float
    drops: int


@dataclass(frozen=True)
class OutputInfo:
    name: str
    uid: str
    transport: str
    latency_ms: float


@dataclass(frozen=True)
class CaptureStatus:
    state: Literal["starting", "running", "no-audio", "restarting", "stopped"]
    message: str = ""
    output: OutputInfo | None = None


def _exact(stream: BinaryIO, count: int, allow_empty: bool = False) -> bytes | None:
    data = bytearray()
    while len(data) < count:
        part = stream.read(count - len(data))
        if not part:
            if allow_empty and not data:
                return None
            raise ProtocolError("truncated message")
        data.extend(part)
    return bytes(data)


def read_messages(stream: BinaryIO) -> Iterator[AudioBlock | dict]:
    while True:
        header = _exact(stream, HEADER.size, allow_empty=True)
        if header is None:
            return
        kind, length = HEADER.unpack(header)
        if length > MAX_PAYLOAD:
            raise ProtocolError(f"oversize message ({length} bytes)")
        payload = _exact(stream, length)
        if kind == b"A":
            if length < AUDIO_HEADER.size or (length - AUDIO_HEADER.size) % 8:
                raise ProtocolError("malformed audio message")
            host_time, drops = AUDIO_HEADER.unpack_from(payload)
            samples = np.frombuffer(payload, "<f4", offset=AUDIO_HEADER.size).reshape(-1, 2).astype(np.float32)
            yield AudioBlock(samples, host_time, drops)
        elif kind == b"J":
            try:
                obj = json.loads(payload)
            except ValueError as error:
                raise ProtocolError(f"bad event JSON: {error}") from None
            if not isinstance(obj, dict) or "event" not in obj:
                raise ProtocolError("event without a type")
            yield obj
        else:
            raise ProtocolError(f"unknown message type {kind!r}")


class CaptureClient:
    """Runs the helper, forwards audio and output-device events, restarts it with backoff."""

    def __init__(self, cmd: list[str], on_audio: Callable[[AudioBlock], None],
                 on_output: Callable[[OutputInfo], None], on_status: Callable[[CaptureStatus], None]):
        self.cmd = list(cmd)
        self.on_audio = on_audio
        self.on_output = on_output
        self.on_status = on_status
        self.launches = 0
        self.pid: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._output: OutputInfo | None = None
        self._state = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._supervise, name="capture-supervisor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._terminate()
        if self._thread is not None:
            self._thread.join(timeout=6)
        self._status("stopped")

    def _status(self, state: str, message: str = "") -> None:
        if (state, message) == self._state and state != "stopped":
            return
        self._state = (state, message)
        self.on_status(CaptureStatus(state, message, self._output))

    def _terminate(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def _supervise(self) -> None:
        failures = 0
        while not self._stop.is_set():
            started = time.monotonic()
            message = self._run_once()
            if self._stop.is_set():
                break
            if time.monotonic() - started >= HEALTHY_RESET_S:
                failures = 0
            delay = BACKOFF_S[min(failures, len(BACKOFF_S) - 1)]
            failures += 1
            log.warning("Capture helper stopped: %s; restarting in %.1f s", message, delay)
            self._status("restarting", message)
            self._stop.wait(delay)

    def _run_once(self) -> str:
        self._status("starting")
        try:
            process = subprocess.Popen(self.cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, bufsize=0)
        except OSError as error:
            return f"Cannot start capture helper: {error}"
        self._process = process
        self.pid = process.pid
        self.launches += 1
        shared = {"ready_at": None, "last_audio": None, "error": None}
        lock = threading.Lock()
        threading.Thread(target=self._drain_stderr, args=(process,), daemon=True).start()
        reader = threading.Thread(target=self._read, args=(process, shared, lock), daemon=True)
        reader.start()
        while reader.is_alive() and not self._stop.is_set():
            reader.join(timeout=.1)
            with lock:
                ready_at, last_audio = shared["ready_at"], shared["last_audio"]
            now = time.monotonic()
            reference = last_audio if last_audio is not None else ready_at
            if reference is not None and now - reference > NO_AUDIO_AFTER_S:
                self._status("no-audio", NO_AUDIO_MESSAGE)
        self._terminate()
        reader.join(timeout=1)
        code = process.poll()
        with lock:
            error = shared["error"]
        return error or f"capture helper exited with code {code}"

    def _read(self, process: subprocess.Popen, shared: dict, lock: threading.Lock) -> None:
        try:
            for item in read_messages(process.stdout):
                if isinstance(item, AudioBlock):
                    with lock:
                        shared["last_audio"] = time.monotonic()
                    self._status("running")
                    self.on_audio(item)
                elif item.get("event") == "ready":
                    with lock:
                        shared["ready_at"] = time.monotonic()
                elif item.get("event") == "output":
                    self._output = OutputInfo(str(item.get("name", "")), str(item.get("uid", "")),
                                              str(item.get("transport", "")), float(item.get("latency_ms", 0.0)))
                    self.on_output(self._output)
                    if self._state is not None:
                        self.on_status(CaptureStatus(self._state[0], self._state[1], self._output))
                elif item.get("event") == "error":
                    with lock:
                        shared["error"] = str(item.get("message", "capture error"))
        except ProtocolError as error:
            with lock:
                shared["error"] = f"capture protocol error: {error}"
        except (OSError, ValueError):
            pass

    @staticmethod
    def _drain_stderr(process: subprocess.Popen) -> None:
        for line in process.stderr:
            text = line.decode("utf8", "replace").rstrip()
            if text:
                log.info("capture: %s", text)
