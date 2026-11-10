"""Continuous, auto-gained levels per band and per layer, plus a silence flag."""
from __future__ import annotations

import math

import numpy as np

from .frontend import Spectrum, band, level_dbfs

# name -> (source, low Hz, high Hz); "mag" is the whole mix, "perc"/"harm" its causal split.
LEVEL_BANDS = {
    "sub": ("mag", 20, 60),
    "bass": ("mag", 60, 250),
    "lowmid": ("mag", 250, 500),
    "mid": ("mag", 500, 2000),
    "presence": ("mag", 2000, 6000),
    "air": ("mag", 6000, 16000),
    "drums": ("perc", 20, 20000),
    "bassline": ("harm", 30, 250),
    "vocals": ("harm", 300, 3400),
    "energy": ("mag", 20, 20000),
}
LEVELS = tuple(LEVEL_BANDS)
RANGE_DB = 40.0             # a level 40 dB under its recent peak reads as 0
RELEASE_DB = 6.0 / 1000     # recent peak falls 6 dB per 10 s (per 10 ms hop)
SILENCE_DBFS = -60.0


class Levels:
    def __init__(self):
        self.masks = {name: band(lo, hi) for name, (_, lo, hi) in LEVEL_BANDS.items()}
        self.peaks = {name: -math.inf for name in LEVELS}

    def step(self, s: Spectrum, hop: np.ndarray) -> tuple[dict[str, float], bool]:
        values = {}
        for name, (source, _, _) in LEVEL_BANDS.items():
            db = level_dbfs(getattr(s, source), self.masks[name])
            self.peaks[name] = max(db, self.peaks[name] - RELEASE_DB)
            values[name] = float(min(1.0, max(0.0, (db - (self.peaks[name] - RANGE_DB)) / RANGE_DB)))
        rms = float(np.sqrt(np.mean(np.square(hop, dtype=np.float64))))
        silent = rms <= 0 or 20 * math.log10(rms) < SILENCE_DBFS
        return values, silent
