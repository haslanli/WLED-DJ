"""When lights should go out, relative to the audio they were computed from.

A sample captured at audio time t is heard at t + output latency. Light leaves the Mac about
NETWORK_LED_MS before it is seen. So a frame should be sent at t + offset, with
offset = (output latency + trim - NETWORK_LED_MS). If processing (P) already took longer than
that, the lights are late by `lead` and beat-locked elements anticipate by that much."""
from __future__ import annotations

from collections import deque
import statistics

NETWORK_LED_MS = 10.0
TRIM_LIMIT_MS = 150.0


class SyncModel:
    def __init__(self, ramp_s: float = 1.0):
        self.ramp_s = ramp_s
        self.latency_from = self.latency_to = 0.0
        self.ramp_start = -1e9
        self.trim_ms = 0.0
        self.processing: deque[float] = deque(maxlen=200)

    def _latency_ms(self, now: float) -> float:
        progress = min(1.0, max(0.0, (now - self.ramp_start) / self.ramp_s)) if self.ramp_s > 0 else 1.0
        return self.latency_from + (self.latency_to - self.latency_from) * progress

    def set_output(self, latency_ms: float, now: float) -> None:
        self.latency_from = self._latency_ms(now)
        self.latency_to = float(latency_ms)
        self.ramp_start = now

    def set_trim(self, ms: float) -> None:
        self.trim_ms = min(TRIM_LIMIT_MS, max(-TRIM_LIMIT_MS, float(ms)))

    def observe_processing(self, ms: float) -> None:
        self.processing.append(float(ms))

    @property
    def processing_ms(self) -> float:
        return statistics.median(self.processing) if self.processing else 0.0

    def offset(self, now: float) -> float:
        return (self._latency_ms(now) + self.trim_ms - NETWORK_LED_MS) / 1000

    def lead(self, now: float) -> float:
        return max(0.0, self.processing_ms / 1000 - self.offset(now))

    def delay(self, now: float) -> float:
        return max(0.0, self.offset(now) - self.processing_ms / 1000)
