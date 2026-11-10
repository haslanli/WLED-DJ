"""Final stage before output: master level, handback fade, blackout, the flash limit, gamma, float -> uint8.

A look's values are linear light to the power 1 / 2.2. The flash limit is judged and applied on linear light,
after master, fade and blackout, so it counts what the lights will really do; the preview shows the same frame.

gamma is 1.0 when WLED already gamma-corrects realtime pixels (its default), else 2.2."""
from __future__ import annotations

import numpy as np

from .flash import FlashLimiter

LIGHT = 2.2                          # look value -> linear light


class Compositor:
    def __init__(self, gamma: float = 2.2, flash_limit: int = 3):
        self.gamma = gamma
        self.master = 1.0
        self.fade = 1.0
        self.blackout = False
        self.limiter = FlashLimiter(flash_limit)
        self._screen: np.ndarray | None = None

    @property
    def flash_limit(self) -> int:
        """Large flashes allowed to start in any one second; 0 is no limit."""
        return self.limiter.limit

    @flash_limit.setter
    def flash_limit(self, limit: int) -> None:
        self.limiter.limit = limit

    @property
    def held(self) -> int:
        """Frames the flash limit had to dim so far."""
        return self.limiter.held

    def compose(self, rgb: np.ndarray, t: float) -> np.ndarray:
        """One frame of a look, at the features' time t, as the bytes for the LEDs."""
        if self.blackout:
            level = np.zeros(rgb.shape, np.float32)
        else:
            level = np.clip(rgb * (self.master * self.fade), 0.0, 1.0)
        linear = level.astype(np.float64) ** LIGHT
        limited = self.limiter.step(linear, t)
        if limited is not linear:                        # held: the frame the lights get is dimmer than the look's
            level = limited ** (1 / LIGHT)
        self._screen = np.round(level * 255).astype(np.uint8)
        return self._screen if self.gamma == 1.0 else np.round(level ** self.gamma * 255).astype(np.uint8)

    def preview(self) -> np.ndarray | None:
        """What the studio draws: the frame just composed, at the lights' levels before LED gamma (a screen
        applies its own). None before the first frame."""
        return self._screen
