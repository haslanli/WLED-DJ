"""What a flash is, and three things built on that: a meter that counts large flashes, a limiter that holds
any picture to a number of them a second, and the budget a look asks before it makes one.

The definitions follow WCAG 2.3.1, on linear light:
- A pixel is "up" while its relative luminance is a tenth or more above its lowest of the last second (and that
  low is under 0.8), or while it is saturated red (red 80% or more of its light) and its R - G - B is 0.0625
  or more above its lowest of the last second, whatever its luminance.
- A large flash starts on the frame the share of pixels that are up reaches a quarter from below. A thin
  strip is small in the field of view, but in a dark room it lights the room, so it is judged as a large area.
- At most `limit` large flashes may start in any one second.

A rise that stays is one start, not a flash for ever: after a second the low has caught up with it."""
from __future__ import annotations

from collections import deque

import numpy as np

LARGE = .25
RISE, LOW_CEILING = .1, .8
RED_SHARE, RED_RISE = .8, .0625
WINDOW_S = 1.0
_LUMA = np.array([.2126, .7152, .0722])
_ROWS = 256                          # frames remembered: more than a second's at any rate the engine runs
_UNDER = .97                         # a held pixel sits this far under its threshold: 8-bit rounding must not lift it over


def luminance(linear) -> np.ndarray:
    """Relative luminance of (..., 3) linear light."""
    return np.asarray(linear, np.float64) @ _LUMA


def _measure(linear: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per pixel: luminance, R - G - B, and whether it is saturated red."""
    total = linear.sum(axis=1)
    return luminance(linear), 2 * linear[:, 0] - total, (total > 0) & (linear[:, 0] >= RED_SHARE * total)


class _Window:
    """Every pixel's luminance and R - G - B over the last second of frames."""

    def __init__(self, pixels: int):
        self.light, self.red = np.zeros((_ROWS, pixels)), np.zeros((_ROWS, pixels))
        self.stamp = np.full(_ROWS, -np.inf)
        self.next = 0

    def lows(self, t: float) -> tuple[np.ndarray, np.ndarray] | None:
        recent = self.stamp > t - WINDOW_S
        if not recent.any():
            return None
        return self.light[recent].min(axis=0), self.red[recent].min(axis=0)

    def add(self, light: np.ndarray, red: np.ndarray, t: float) -> None:
        self.light[self.next], self.red[self.next], self.stamp[self.next] = light, red, t
        self.next = (self.next + 1) % _ROWS


class FlashMeter:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.window: _Window | None = None
        self.large = False                               # a large flash is under way
        self.starts: deque[float] = deque()              # when the large flashes of the last second started

    def judge(self, linear: np.ndarray, t: float) -> dict[str, np.ndarray]:
        """Which pixels of this frame are up, against the frames already taken in. Takes nothing in itself."""
        if self.window is None or self.window.light.shape[1] != len(linear):
            self.reset()
            self.window = _Window(len(linear))
        light, red, saturated = _measure(linear)
        lows = self.window.lows(t)
        low_light, low_red = lows if lows is not None else (light, red)
        bright = (light - low_light >= RISE) & (low_light < LOW_CEILING)
        redder = saturated & (red - low_red >= RED_RISE)
        return dict(light=light, red=red, low_light=low_light, low_red=low_red, bright=bright, redder=redder,
                    up=bright | redder)

    def recent(self, t: float) -> int:
        """How many large flashes started in the last second."""
        while self.starts and t - self.starts[0] >= WINDOW_S:
            self.starts.popleft()
        return len(self.starts)

    def take(self, seen: dict[str, np.ndarray], t: float) -> bool:
        """Take a judged frame in. True if a large flash starts with it."""
        self.recent(t)
        large = bool(seen["up"].mean() >= LARGE)
        started = large and not self.large
        self.large = large
        if started:
            self.starts.append(t)
        self.window.add(seen["light"], seen["red"], t)
        return started

    def step(self, linear, t: float) -> bool:
        """One frame of (pixels, 3) linear light at time t. True if a large flash starts on it."""
        linear = np.asarray(linear, np.float64)
        return self.take(self.judge(linear, t), t)


class FlashLimiter:
    """The guarantee: whatever comes in, at most `limit` large flashes a second go out. 0 switches it off."""

    def __init__(self, limit: int = 3):
        self.limit = limit
        self.meter = FlashMeter()
        self.held = 0                                    # frames it had to change

    def step(self, linear, t: float) -> np.ndarray:
        if self.limit <= 0:
            self.meter.reset()
            return linear
        meter = self.meter
        frame = np.asarray(linear, np.float64)
        seen = meter.judge(frame, t)
        if seen["up"].mean() >= LARGE and not meter.large and meter.recent(t) >= self.limit:
            # This would be one flash too many: every pixel that is up stays just under what makes it a flash,
            # on this frame and on the next ones, until a second has room again.
            factor = np.ones(len(frame))
            bright, redder = seen["bright"], seen["redder"]
            factor[bright] = (seen["low_light"][bright] + RISE) * _UNDER / seen["light"][bright]
            factor[redder] = np.minimum(factor[redder], (seen["low_red"][redder] + RED_RISE) * _UNDER / seen["red"][redder])
            frame = frame * np.clip(factor, 0.0, 1.0)[:, None]
            self.held += 1
            light, red, _ = _measure(frame)
            meter.window.add(light, red, t)
            return frame
        meter.take(seen, t)
        return linear


class FlashBudget:
    """The manners: a look asks before a large gesture, and draws it small when the answer is no."""

    def __init__(self, limit: int = 3):
        self.limit = limit
        self.granted: deque[float] = deque()

    def ask(self, t: float) -> bool:
        while self.granted and not 0 <= t - self.granted[0] < WINDOW_S:
            self.granted.popleft()
        if self.limit <= 0:
            return True
        if len(self.granted) >= self.limit:
            return False
        self.granted.append(t)
        return True


def largest_in_a_second(frames_linear, fps: float) -> int:
    """The most large flashes that start inside any one second of (T, pixels, 3) linear-light frames."""
    meter = FlashMeter()
    starts = [i for i, frame in enumerate(frames_linear) if meter.step(frame, i / fps)]
    return max((sum(1 for other in starts if start <= other < start + fps) for start in starts), default=0)
