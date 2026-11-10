"""The lead voice: where its pitch centre sits and how it moves.

The lead is the vocals while a voice is present, otherwise whatever carries the tune in the "other" layer.
Its centre is the middle of its harmonic spectrum on a pitch scale (200-6000 Hz). A look does not need hertz,
it needs a place: `position` puts the centre inside the range the lead itself has covered lately, and
`velocity` says how fast it travels there. A filter sweep or a glissando is a slide across that range."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from .frontend import HOP_S, NFFT, Spectrum, band
from .layers import Layers

LOW_HZ, HIGH_HZ = 200.0, 6000.0
CENTRE_S = .2                        # smoothing of the centre (the 23 ms frames it is read from jump about)
VELOCITY_S = .1
TRAVEL_S, TRAVEL_SMOOTH_S = 4.0, .5   # travel: how far the position, smoothed over half a second, went in 4 s
RANGE_S = 30.0                       # the range's ends ease toward the lead this slowly; they stretch at once to hold it
NARROW = 1 / 6                       # octaves (two semitones): a range under this is no range, and the lead sits mid-way
QUIET_DB = -50.0                     # under this the lead is not there
METER_FLOOR_DB = -60.0


@dataclass(frozen=True, slots=True)
class Voice:
    position: float = .5             # 0-1: where the lead's pitch centre sits in the range it has covered lately
    velocity: float = 0.0            # position per second; negative is falling
    level: float = 0.0               # 0-1 from the lead's 50 ms level: -60 dB is 0, 0 dB is 1
    vocal: bool = False              # the lead is the vocals layer
    hz: float = 0.0                  # its pitch centre, for tests and the timeline; looks read position


def _follow(seconds: float) -> float:
    return 1 - math.exp(-HOP_S / seconds)


class LeadTracker:
    def __init__(self, rate: int):
        self.mask = band(LOW_HZ, HIGH_HZ, rate)
        self.pitch = np.log2(np.fft.rfftfreq(NFFT, 1 / rate)[self.mask])
        self.centre: float | None = None             # log2 Hz, smoothed
        self.low = self.high = 0.0
        self.voice = Voice()
        self.steady = .5                             # the position, smoothed
        self.steadies: deque[float] = deque(maxlen=round(TRAVEL_S / HOP_S) + 1)
        self.travel = 0.0                            # how far the lead went in its range in the last 4 s: the way made, not the wobble

    def step(self, vocals: Spectrum, other: Spectrum, layers: Layers) -> Voice:
        """One hop: the spectra of the vocals and other layers, and the layers' levels and presence."""
        vocal = layers.vocals.present
        layer, spectrum = (layers.vocals, vocals) if vocal else (layers.other, other)
        power = spectrum.harm[self.mask].astype(np.float64) ** 2
        total = float(power.sum())
        if layer.fast_db < QUIET_DB or total <= 0:
            self.voice = Voice(self.voice.position, 0.0, 0.0, vocal, self.voice.hz)
            self.steadies.clear()                    # a lead that comes back has not travelled
            self.travel = 0.0
            return self.voice
        centre = float(np.dot(self.pitch, power)) / total
        if self.centre is None:
            self.centre = self.low = self.high = centre
        self.centre += (centre - self.centre) * _follow(CENTRE_S)
        ease = _follow(RANGE_S)
        self.low = min(self.centre, self.low + (self.centre - self.low) * ease)
        self.high = max(self.centre, self.high + (self.centre - self.high) * ease)
        span = self.high - self.low
        own = min(1.0, max(0.0, span / (NARROW / 2) - 1))       # 0 under one semitone of range, 1 from two
        position = .5 + ((self.centre - self.low) / span - .5) * own if span > 0 else .5
        velocity = self.voice.velocity + ((position - self.voice.position) / HOP_S - self.voice.velocity) * _follow(VELOCITY_S)
        level = min(1.0, max(0.0, 1 - layer.fast_db / METER_FLOOR_DB))
        self.steady = position if not self.steadies else self.steady + (position - self.steady) * _follow(TRAVEL_SMOOTH_S)
        self.steadies.append(self.steady)
        self.travel = abs(self.steady - self.steadies[0])
        self.voice = Voice(position, velocity, level, vocal, 2.0 ** self.centre)
        return self.voice
