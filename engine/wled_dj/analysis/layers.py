"""The music's four layers (drums, bass, vocals, other): how loud each is and whether it is playing.

Levels are mean power in dB relative to a full-scale sine. A layer is present when its 400 ms level is
within a threshold of the mix's and above -50 dB; it enters after 150 ms of that and leaves after a
second below, so a gap between two kicks is not an exit."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .frontend import FULL_SCALE_ENERGY, HOP_S, Spectrum, band

LAYERS = ("drums", "bass", "vocals", "other")
# How far under the mix a layer may be and still count as playing: chosen by bench/score_layers.py --tune
# on 25 MUSDB18 test songs for the best balanced accuracy, and scored on the other 25. Bass from stems is
# the exception: -9.5 and -15 score the same (0.78 and 0.79), and -15 misses far fewer real bass passages
# (9% against 25%) at the cost of more false ones, which is the cheaper error for a light show.
PRESENT_REL_DB = {"stems": {"drums": -13.5, "bass": -15.0, "vocals": -13.5, "other": -8.0},
                  "bands": {"drums": -6.0, "bass": -5.0, "vocals": -9.0, "other": -5.0}}
PRESENT_ABS_DB = -50.0
ENTER_S, LEAVE_S, LEAVE_MARGIN_DB = .15, 1.0, 3.0
LEVEL_S, FAST_S, KICK_S = .4, .05, 1.0
FLOOR_DB = -120.0
_BASS, _VOCALS = band(0, 250), band(300, 3400)


@dataclass(frozen=True, slots=True)
class Layer:
    level_db: float      # mean power over 400 ms
    fast_db: float       # over 50 ms
    rel_db: float        # level_db minus the mix's
    present: bool
    since_s: float       # how long `present` has held its value


@dataclass(frozen=True, slots=True)
class Layers:
    drums: Layer
    bass: Layer
    vocals: Layer
    other: Layer
    kick_weight_db: float    # the drums' 35-130 Hz band over one second
    source: str              # "stems" or "bands"

    def __getitem__(self, name: str) -> Layer:
        return getattr(self, name)


def power_db(power: float) -> float:
    """dB relative to a full-scale sine (mean power 0.5), never under FLOOR_DB."""
    return max(FLOOR_DB, 10 * math.log10(power / .5 + 1e-12))


class Mean:
    """Mean of the last `seconds` of per-hop values. Until that long has passed the missing values count as
    zero, or, with `so_far`, the mean is of the values there are."""

    def __init__(self, seconds: float, so_far: bool = False):
        self.values: deque[float] = deque(maxlen=max(1, round(seconds / HOP_S)))
        self.total = 0.0
        self.so_far = so_far

    def add(self, value: float) -> float:
        if len(self.values) == self.values.maxlen:
            self.total -= self.values[0]
        self.values.append(value)
        self.total += value
        return max(0.0, self.total) / (len(self.values) if self.so_far else self.values.maxlen)


class _Presence:
    def __init__(self):
        self.present = False
        self.above_s = self.below_s = self.since_s = 0.0

    def step(self, above: bool, below: bool) -> None:
        self.above_s = self.above_s + HOP_S if above else 0.0
        self.below_s = self.below_s + HOP_S if below else 0.0
        self.since_s += HOP_S
        if (not self.present and self.above_s >= ENTER_S - 1e-9) or (self.present and self.below_s >= LEAVE_S - 1e-9):
            self.present = not self.present
            self.since_s = 0.0


class LayerTracker:
    def __init__(self, source: str):
        self.source = source
        self.thresholds = PRESENT_REL_DB[source]
        self.level = {name: Mean(LEVEL_S) for name in LAYERS}
        self.fast = {name: Mean(FAST_S) for name in LAYERS}
        self.mix = Mean(LEVEL_S)
        self.kick = Mean(KICK_S)
        self.presence = {name: _Presence() for name in LAYERS}

    def step(self, power: Mapping[str, float], mix_power: float, kick_power: float) -> Layers:
        """One 10 ms hop: the mean power of each layer, of the mix and of the drums' low band."""
        mix_db = power_db(self.mix.add(mix_power))
        layers = {}
        for name in LAYERS:
            level_db = power_db(self.level[name].add(power[name]))
            rel_db = level_db - mix_db
            threshold = self.thresholds[name]
            state = self.presence[name]
            state.step(above=rel_db > threshold and level_db > PRESENT_ABS_DB,
                       below=rel_db < threshold - LEAVE_MARGIN_DB or level_db < PRESENT_ABS_DB - LEAVE_MARGIN_DB)
            layers[name] = Layer(level_db, power_db(self.fast[name].add(power[name])), rel_db, state.present, state.since_s)
        return Layers(**layers, kick_weight_db=power_db(self.kick.add(kick_power)), source=self.source)


def bands_power(spectrum: Spectrum) -> tuple[dict[str, float], float, float]:
    """Layer powers read from the mix alone: drums = the percussive part, bass = the harmonic part under
    250 Hz, vocals = the harmonic part 300-3400 Hz, other = the rest. Returns (layers, mix, kick)."""
    scale = .5 / FULL_SCALE_ENERGY
    harmonic = spectrum.harm ** 2
    total = float((spectrum.mag ** 2).sum()) * scale
    power = {"drums": float((spectrum.perc ** 2).sum()) * scale,
             "bass": float(harmonic[_BASS].sum()) * scale,
             "vocals": float(harmonic[_VOCALS].sum()) * scale}
    power["other"] = max(0.0, total - sum(power.values()))
    return power, total, spectrum.low_power
