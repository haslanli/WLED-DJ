"""The engine's side of layer separation: runs the stem helper, feeds it audio, and hands back stems
on the engine's 10 ms grid.

After engine hop n is pushed, the model has been given every complete 128-sample hop of the 44.1 kHz
audio so far, and it answers one hop behind. So what is certain to exist is everything up to 256 samples
(5.8 ms) before the end of hop n. Stem hop n is therefore the 441 samples ending there: it trails the mix
by `latency_s`, and needs no audio the engine does not have yet.

The engine never waits here longer than it asks to. A helper that is late more often than not, stops
reading or exits is killed, and started again after a pause that grows to a minute while trouble lasts."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import os
import subprocess
import threading
import time

import numpy as np

from ..analysis.resample import Resampler
from . import protocol

log = logging.getLogger(__name__)

STEM_HOP = 441                       # 10 ms at 44.1 kHz
# A late hop is covered where it happens (its events come from the mix). Only a helper that is late more
# often than not over ten seconds is paused: one stall leaves a run of 50 to 250 late hops behind it.
LATE_WINDOW, LATE_LIMIT = 1000, .5
PAUSES_S = (10, 20, 40, 60)
HEALTHY_S = 60.0                     # this long without a pause and the pauses start short again


@dataclass(frozen=True)
class StemHop:
    index: int               # the engine hop it belongs to (the value push returned)
    t: float                 # host time of its last sample
    drums: np.ndarray        # (441,) float32
    bass: np.ndarray
    vocals: np.ndarray
    other: np.ndarray        # the mix minus the three
    mix: np.ndarray          # the same samples of the mono mix


class StemLink:
    def __init__(self, cmd: list[str], clock=time.monotonic):
        self.cmd = list(cmd)
        self.clock = clock
        self.state = "off"                       # off | starting | on | paused
        self.latency_s = Resampler().delay_s + protocol.LATENCY / protocol.RATE
        self.frames_sent = 0                     # model hops sent to the helper since it started
        self._lock = threading.Lock()
        self._arrived = threading.Condition(self._lock)
        self._process: subprocess.Popen | None = None
        self._generation = 0
        self._replies: list[np.ndarray] = []
        self._received = 0
        self._judged: deque[bool] = deque(maxlen=LATE_WINDOW)
        self._pauses = 0
        self._resume_at = 0.0
        self._on_since = 0.0
        self._reset_stream()

    # ---- state -----------------------------------------------------------------------------
    @property
    def pid(self) -> int | None:
        process = self._process
        return None if process is None else process.pid

    @property
    def late_share(self) -> float:
        """Late hops as a share of the last ten seconds (hops not yet pushed count as on time)."""
        return list(self._judged).count(False) / LATE_WINDOW

    def start(self) -> None:
        with self._lock:
            if self.state in ("starting", "on"):
                return
            self._pauses = 0
            self._spawn()

    def stop(self) -> None:
        with self._lock:
            process = self._end_process()
            self.state = "off"
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def _reset_stream(self) -> None:
        self._resampler = Resampler()
        self._unsent = np.zeros((0, 2), np.float32)
        # Both buffers start 256 samples before the stream, so stem hop 0 is their first 441 samples.
        self._mix = np.zeros(protocol.LATENCY, np.float32)
        self._stems = np.zeros((3, protocol.LATENCY), np.float32)
        self._times: dict[int, float] = {}
        self._awaiting: deque[int] = deque()
        self._hops = self._next = 0
        self.frames_sent = 0
        self._judged.clear()

    def _spawn(self) -> None:
        self._generation += 1
        self._replies, self._received = [], 0
        self._reset_stream()
        try:
            process = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, bufsize=0)
        except OSError as error:
            log.warning("Cannot start the stem helper: %s", error)
            self._pause("it could not be started")
            return
        os.set_blocking(process.stdin.fileno(), False)
        self._process = process
        self.state = "starting"
        threading.Thread(target=self._read, args=(process, self._generation), name="stems-reader", daemon=True).start()
        threading.Thread(target=self._drain, args=(process,), name="stems-log", daemon=True).start()

    def _end_process(self) -> subprocess.Popen | None:
        """Forget the helper (its reader thread stops counting). The caller ends the process."""
        self._generation += 1
        process, self._process = self._process, None
        return process

    def _pause(self, reason: str) -> None:
        process = self._end_process()
        if process is not None:
            process.kill()
            threading.Thread(target=process.wait, daemon=True).start()       # reap it without waiting here
        now = self.clock()
        if self.state == "on" and now - self._on_since >= HEALTHY_S:
            self._pauses = 0
        delay = PAUSES_S[min(self._pauses, len(PAUSES_S) - 1)]
        self._pauses += 1
        self._resume_at = now + delay
        self.state = "paused"
        log.warning("Layer separation paused for %d s: %s", delay, reason)

    def _read(self, process: subprocess.Popen, generation: int) -> None:
        while True:
            try:
                data = protocol.read_exact(process.stdout, protocol.OUT_BYTES)
            except (OSError, ValueError):
                data = None
            with self._arrived:
                if generation != self._generation:
                    return
                if data is None:
                    self._pause("the helper stopped")
                    self._arrived.notify_all()
                    return
                index, stems = protocol.unpack_out(data)
                if index == protocol.READY:
                    if self.state == "starting":
                        self.state = "on"
                        self._on_since = self.clock()
                    continue
                if index >= 1:                       # frame 0 answers for audio before the stream
                    self._replies.append(stems)
                self._received = index + 1
                self._arrived.notify_all()

    @staticmethod
    def _drain(process: subprocess.Popen) -> None:
        for line in process.stderr:
            text = line.decode("utf8", "replace").rstrip()
            if text:
                log.info("stems: %s", text)

    # ---- audio -----------------------------------------------------------------------------
    def push(self, hop48: np.ndarray, t_end: float) -> int | None:
        """One 480 x 2 hop of the mix at 48 kHz and the host time of its last sample. Never waits.
        Returns the hop's index, or None while separation is not running."""
        with self._lock:
            if self.state == "paused" and self.clock() >= self._resume_at:
                self._spawn()
            if self.state != "on":
                return None
            index = self._hops
            self._hops += 1
            self._times[index] = t_end
            self._awaiting.append(index)
            audio = self._resampler.push(hop48)
            self._mix = np.concatenate((self._mix, audio.mean(axis=1)))
            self._unsent = np.concatenate((self._unsent, audio))
            fd = self._process.stdin.fileno()
            while len(self._unsent) >= protocol.HOP:
                frame = protocol.pack_in(self.frames_sent, self._unsent[:protocol.HOP].T)
                try:
                    written = os.write(fd, frame)
                except OSError:                      # the pipe is full (BlockingIOError) or closed
                    written = 0
                if written != len(frame):
                    self._pause("the helper is not reading")
                    return None
                self.frames_sent += 1
                self._unsent = self._unsent[protocol.HOP:]
            return index

    def collect(self, wait_s: float) -> list[StemHop]:
        """Every stem hop completed since the last call, oldest first. Waits up to wait_s for the answer
        to the last frame sent, which completes the hop pushed last."""
        deadline = time.monotonic() + wait_s
        with self._arrived:
            while self.state == "on" and self._received < self.frames_sent:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._arrived.wait(remaining)
            if self.state != "on":
                return []
            replies, self._replies = self._replies, []
            if replies:
                self._stems = np.concatenate([self._stems, *replies], axis=1)
            hops = []
            while self._stems.shape[1] >= STEM_HOP and self._next in self._times:
                drums, bass, vocals = self._stems[:, :STEM_HOP]
                mix = self._mix[:STEM_HOP]
                hops.append(StemHop(self._next, self._times.pop(self._next) - self.latency_s,
                                    drums, bass, vocals, mix - drums - bass - vocals, mix))
                self._stems, self._mix = self._stems[:, STEM_HOP:], self._mix[STEM_HOP:]
                self._next += 1
            returned = {hop.index for hop in hops}
            while self._awaiting:
                self._judged.append(self._awaiting.popleft() in returned)
            if self.late_share > LATE_LIMIT:
                self._pause(f"{self.late_share:.0%} of the last ten seconds' answers were late")
                return []
            return hops
