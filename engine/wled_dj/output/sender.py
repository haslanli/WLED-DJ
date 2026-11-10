"""Timed frame sender: never early, at most one frame per 1/fps, always the newest frame that is due."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import bisect
import logging
import statistics
import threading
import time
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

ERROR_PAUSE_S = .1
IDLE_WAIT_S = .05


@dataclass(order=True)
class TimedFrame:
    due: float
    audio_t: float = field(compare=False)
    rgb: np.ndarray = field(compare=False)


class Sender:
    def __init__(self, transmit: Callable[[np.ndarray], bool], fps: float, clock=time.monotonic):
        self.transmit = transmit
        self.fps = fps
        self.clock = clock
        self.active = True
        self._frames: list[TimedFrame] = []
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._next_allowed = -1e18
        self._pause_until = -1e18
        self._last_send: float | None = None
        self._last_audio_t = -1e18             # never send audio older than what already went out
        self._sent = self._skipped = self._errors = 0
        self._latencies: deque[float] = deque(maxlen=200)
        self._intervals: deque[float] = deque(maxlen=500)
        self._running = False
        self._thread: threading.Thread | None = None

    def submit(self, frame: TimedFrame) -> None:
        with self._wake:
            if frame.audio_t <= self._last_audio_t:
                self._skipped += 1
                return
            bisect.insort(self._frames, frame)
            if len(self._frames) > 1000:        # sender stalled; keep memory bounded
                self._skipped += len(self._frames) - 1000
                del self._frames[:-1000]
            self._wake.notify()

    def pump(self, now: float) -> float:
        """Make at most one send decision at time `now`; return when to look again."""
        with self._lock:
            if not self.active:
                self._frames.clear()
                return now + IDLE_WAIT_S
            due = bisect.bisect_right(self._frames, TimedFrame(now, 0.0, None))
            ready_at = max(self._next_allowed, self._pause_until)
            if due == 0 or now < ready_at:
                upcoming = self._frames[0].due if self._frames else now + IDLE_WAIT_S
                return max(upcoming, ready_at)
            frame = max(self._frames[:due], key=lambda f: f.audio_t)
            kept = [f for f in self._frames[due:] if f.audio_t > frame.audio_t]
            self._skipped += len(self._frames) - len(kept) - 1
            self._frames = kept
            self._last_audio_t = frame.audio_t
        try:
            ok = bool(self.transmit(frame.rgb))
        except Exception:
            log.exception("Frame transmit failed")
            ok = False
        with self._lock:
            if ok:
                self._sent += 1
                self._latencies.append((now - frame.audio_t) * 1000)
                if self._last_send is not None:
                    self._intervals.append((now - self._last_send) * 1000)
                self._last_send = now
                self._next_allowed = now + 1 / self.fps
            else:
                self._errors += 1
                self._pause_until = now + ERROR_PAUSE_S
            return max(self._next_allowed, self._pause_until)

    def stats(self) -> dict:
        with self._lock:
            intervals = sorted(self._intervals)
            p99 = intervals[min(len(intervals) - 1, int(.99 * len(intervals)))] if intervals else 0.0
            return {"sent": self._sent, "skipped": self._skipped, "errors": self._errors,
                    "latency_ms": statistics.median(self._latencies) if self._latencies else 0.0,
                    "interval_p99_ms": p99}

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name="ddp-sender", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._wake:
            self._running = False
            self._wake.notify()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while self._running:
            try:
                wake_at = self.pump(self.clock())
            except Exception:                       # a sender that dies leaves the lights frozen: never die
                log.exception("Sender error")
                wake_at = self.clock() + ERROR_PAUSE_S
            with self._wake:
                if not self._running:
                    break
                wait = wake_at - self.clock()
                if wait > 0:
                    self._wake.wait(min(wait, IDLE_WAIT_S))
