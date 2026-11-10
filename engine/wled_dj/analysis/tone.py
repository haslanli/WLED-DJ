"""The harmony's colour: the pitch classes sounding now, as an angle on the circle of fifths.

A fifth apart is 30 degrees apart, so related chords are neighbours and the same chord always lands on the
same angle, in any octave. Two averages are kept: a slow one (the song's colour) and a fast one (the chord's).
While the sound has no pitch to trust (too quiet, or as flat as noise), both angles hold."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from .frontend import HOP_S
from .layers import power_db

WINDOW = 8192                        # samples: about 5 Hz a bin at 44.1 kHz; half of it misreads chords under middle C
LOW_HZ, HIGH_HZ = 55.0, 2000.0       # where the notes are; above it the partials of one note crowd every class
SLOW_S, FAST_S = 6.0, .7
QUIET_DB = -50.0
FLAT = .3                            # spectral flatness (geometric over arithmetic mean power): noise is 0.56, a chord near 0
SHORT = .1                           # a profile this short on the circle points nowhere
SURE = .2                            # the fast average must be this long for its turning to count as a chord change
FIFTHS = np.exp(2j * np.pi * ((7 * np.arange(12)) % 12) / 12)


@dataclass(frozen=True, slots=True)
class Tone:
    angle: float = 0.0               # the song's colour: degrees 0-360 on the circle of fifths, over about 6 s
    chord: float = 0.0               # the same over about 0.7 s: it moves with the chords
    confidence: float = 0.0          # 0-1: length of the fast average on the circle; 0 while the angles hold


def _degrees(z: complex) -> float:
    return math.degrees(math.atan2(z.imag, z.real)) % 360


class ToneTracker:
    def __init__(self, rate: int):
        freq = np.fft.rfftfreq(WINDOW, 1 / rate)
        self.bins = (freq >= LOW_HZ) & (freq <= HIGH_HZ)
        self.pitch_class = np.mod(np.round(12 * np.log2(freq[self.bins] / 440.0) + 69), 12).astype(int)
        self.window = np.hanning(WINDOW).astype(np.float32)
        self.buffer = np.zeros(WINDOW, np.float32)
        self.slow: np.ndarray | None = None          # running pitch-class profiles; None until the first trusted frame
        self.fast: np.ndarray | None = None
        self.tone = Tone()
        self.chords: deque[float] = deque(maxlen=round(1 / HOP_S) + 1)
        self.chord_step = 0.0                        # degrees the chord colour moved in the last second, while it is sure of itself

    def step(self, harmonic: np.ndarray) -> Tone:
        """One hop (rate / 100 mono samples) of the layers that carry pitch: bass + other + vocals."""
        n = len(harmonic)
        self.buffer[:-n] = self.buffer[n:]
        self.buffer[-n:] = harmonic
        profile = self._profile()
        if profile is None:
            self.tone = Tone(self.tone.angle, self.tone.chord, 0.0)
        else:
            if self.slow is None:
                self.slow, self.fast = profile.copy(), profile.copy()
            self.slow += (profile - self.slow) * (1 - math.exp(-HOP_S / SLOW_S))
            self.fast += (profile - self.fast) * (1 - math.exp(-HOP_S / FAST_S))
            slow, fast = complex(np.dot(self.slow, FIFTHS)), complex(np.dot(self.fast, FIFTHS))
            self.tone = Tone(_degrees(slow), _degrees(fast), min(1.0, abs(fast)))
        self.chords.append(self.tone.chord)
        turned = abs((self.tone.chord - self.chords[0] + 180) % 360 - 180)
        self.chord_step = turned if self.tone.confidence >= SURE else 0.0
        return self.tone

    def _profile(self) -> np.ndarray | None:
        """This frame's power per pitch class, summing to 1, or None when it has no pitch to trust."""
        if power_db(float(np.mean(np.square(self.buffer, dtype=np.float64)))) < QUIET_DB:
            return None
        power = np.abs(np.fft.rfft(self.buffer * self.window))[self.bins].astype(np.float64) ** 2
        total = float(power.sum())
        if total <= 0 or math.exp(float(np.mean(np.log(power + 1e-30)))) / (total / len(power)) > FLAT:
            return None
        profile = np.bincount(self.pitch_class, weights=power, minlength=12) / total
        return profile if abs(np.dot(profile, FIFTHS)) >= SHORT else None
