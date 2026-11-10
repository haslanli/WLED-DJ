"""Calibration look: the whole strip flashes for 60 ms on every detected kick, nothing else."""
from __future__ import annotations

import math

import numpy as np

from ..analysis.features import Features

FLASH_S = .06


class SyncCheck:
    id = "sync-check"
    name = "Sync check"

    def __init__(self, pixels: int):
        self.pixels = pixels
        self.kick_at = -math.inf

    def render(self, f: Features, lead: float) -> np.ndarray:
        if f.kick is not None:
            self.kick_at = f.t
        value = 1.0 if 0 <= f.t - self.kick_at < FLASH_S - 1e-9 else 0.0
        return np.full((self.pixels, 3), value, np.float32)
