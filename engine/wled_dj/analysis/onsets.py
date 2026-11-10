"""Causal onset detectors per instrument class.

Novelty: sum over the band of positive log-magnitude increases. Fires on a rising edge above
median + k * MAD of the previous second, when the band is loud enough, then holds off for the
refractory period."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from .frontend import HOP_S, RATE, Spectrum, band, level_dbfs

HISTORY = 100              # previous novelty values for the adaptive threshold (1 s)
MIN_HISTORY = 10
RELATIVE_GATE_DB = 40.0    # band must be within 40 dB of its own recent peak
ABSOLUTE_GATE_DBFS = -60.0  # and louder than -60 dBFS (rejects silence and room noise)
PEAK_RELEASE_DB = 6.0 / 1000  # recent peak falls 6 dB per 10 s (per 10 ms hop)
MIN_NOVELTY_PER_BIN = 0.1  # floor on the threshold so steady tones never trigger
MIN_RISE_DB = 3.0          # floor for the low-band envelope detector
LOW_RISE_HOPS = 3          # envelope rise is measured against the mean of the previous 30 ms


@dataclass(frozen=True)
class Onset:
    name: str
    strength: float
    pan: float


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    source: Literal["low", "perc", "harm"]
    lo: float
    hi: float
    k: float
    refractory_s: float


# k tuned by bench/score_detection.py --tune on MUSDB18-7 train (F0.5 within 0.05 of best F1); see bench/out/detection.json.
SPECS: tuple[DetectorSpec, ...] = (
    DetectorSpec("kick", "low", 35, 130, 0.5, .09),
    DetectorSpec("snare", "perc", 1000, 5000, 4.5, .07),
    DetectorSpec("hats", "perc", 7000, 16000, 5.75, .05),
    DetectorSpec("bass", "harm", 30, 300, 1.75, .09),
    DetectorSpec("vocal", "harm", 200, 4000, 0.5, .12),
)

# The same detectors read from separated stems: (stem, spec). k tuned by bench/score_layers.py --tune on
# 25 MUSDB18 test songs (most precise within 0.05 of the best F1) and scored on the other 25.
STEM_SPECS: tuple[tuple[str, DetectorSpec], ...] = (
    ("drums", DetectorSpec("kick", "low", 35, 130, 7.5, .09)),
    ("drums", DetectorSpec("snare", "perc", 1000, 5000, 8.5, .07)),
    ("drums", DetectorSpec("hats", "perc", 7000, 16000, 8.0, .05)),
    ("bass", DetectorSpec("bass", "harm", 30, 300, 1.5, .09)),
    ("vocals", DetectorSpec("vocal", "harm", 200, 4000, .5, .12)),
    ("other", DetectorSpec("stab", "harm", 200, 6000, 1.0, .12)),    # chords and synth hits: only stems can tell them from a voice
)


class OnsetDetector:
    def __init__(self, spec: DetectorSpec, rate: int = RATE):
        self.spec = spec
        self.mask = band(spec.lo, spec.hi, rate)
        self.pan = 0.0             # right minus left over their sum in this band, for the latest hop
        self.floor = MIN_RISE_DB if spec.source == "low" else MIN_NOVELTY_PER_BIN * max(1, int(self.mask.sum()))
        self.previous: np.ndarray | None = None
        self.low_history: deque[float] = deque(maxlen=LOW_RISE_HOPS)
        self.history: deque[float] = deque(maxlen=HISTORY)
        self.peak_db = -math.inf
        self.was_above = False
        self.hops_since = math.inf
        self.novelty = 0.0

    def _novelty_and_level(self, s: Spectrum) -> tuple[float, float, float, float]:
        """Returns (novelty, level dBFS, left energy, right energy) for this hop."""
        if self.spec.source == "low":
            power = s.low_power + 1e-12
            novelty = 0.0
            if self.low_history:
                novelty = max(0.0, 10 * math.log10(power / (sum(self.low_history) / len(self.low_history))))
            self.low_history.append(power)
            return novelty, 10 * math.log10(power / 0.5), s.low_left, s.low_right
        source = s.perc if self.spec.source == "perc" else s.harm
        current = np.log1p(1000 * source[self.mask])
        novelty = 0.0 if self.previous is None else float(np.maximum(current - self.previous, 0).sum())
        self.previous = current
        left = float(np.sum(s.mag_left[self.mask] ** 2))
        right = float(np.sum(s.mag_right[self.mask] ** 2))
        return novelty, level_dbfs(s.mag, self.mask), left, right

    def step(self, s: Spectrum) -> Onset | None:
        novelty, level, left, right = self._novelty_and_level(s)
        self.novelty = novelty
        self.pan = (right - left) / (right + left + 1e-12)
        self.hops_since += 1
        self.peak_db = max(level, self.peak_db - PEAK_RELEASE_DB)
        onset = None
        if len(self.history) >= MIN_HISTORY:
            past = np.fromiter(self.history, float)
            median = float(np.median(past))
            mad = float(np.median(np.abs(past - median)))
            threshold = median + max(self.spec.k * mad, self.floor)
            above = (novelty > threshold and level > ABSOLUTE_GATE_DBFS
                     and level > self.peak_db - RELATIVE_GATE_DB)
            if above and not self.was_above and self.hops_since * HOP_S >= self.spec.refractory_s:
                strength = min(1.0, max(0.0, 1 - math.exp(-(novelty - threshold) / threshold)))
                onset = Onset(self.spec.name, strength, self.pan)
                self.hops_since = 0
            self.was_above = above
        self.history.append(novelty)
        return onset
