"""Causal tempo and beat-phase tracking from an onset-strength envelope sampled every 10 ms.

Every 0.25 s: autocorrelate the last 8 s of the (lightly smoothed) envelope over 70-180 BPM with a
log-Gaussian tempo prior centred on 120 BPM, refine the peak by parabolic interpolation, then find
the beat offset with a comb over the last 4 s. Between estimates the phase advances at the current
tempo and is pulled 20% toward each new estimate. The downbeat is the beat position (mod 4) that
has collected the most kick strength."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

STEP = .01                  # seconds per envelope sample
WINDOW = 800                # 8 s of envelope
COMB_WINDOW = 400           # 4 s for the phase comb
MIN_HISTORY = 300           # 3 s before the first estimate
UPDATE_EVERY = 25           # 0.25 s
MIN_BPM, MAX_BPM = 70.0, 180.0
PRIOR_BPM, PRIOR_OCTAVES = 120.0, 1.0
PHASE_PULL = .2
SMOOTH = np.array([.25, .5, .25])


@dataclass(frozen=True)
class BeatState:
    bpm: float = 0.0
    phase: float = 0.0
    bar_phase: float = 0.0
    confidence: float = 0.0
    t: float = 0.0

    def phase_at(self, t_future: float) -> float:
        if self.bpm <= 0:
            return self.phase
        return (self.phase + (t_future - self.t) * self.bpm / 60) % 1.0


def _interp(signal: np.ndarray, index: float) -> float:
    lo = int(math.floor(index))
    if lo < 0 or lo + 1 >= len(signal):
        return 0.0
    frac = index - lo
    return float(signal[lo] * (1 - frac) + signal[lo + 1] * frac)


class BeatTracker:
    def __init__(self):
        self.envelope: deque[float] = deque(maxlen=WINDOW)
        self.steps = 0
        self.bpm = 0.0
        self.phase = 0.0
        self.confidence = 0.0
        self.last_t: float | None = None
        self.estimates: deque[float] = deque(maxlen=4)
        self.beat_index = 0
        self.kick_by_position = np.zeros(4)

    def step(self, t: float, strength: float, kick=None) -> BeatState:
        if self.last_t is not None and self.bpm > 0:
            advanced = self.phase + (t - self.last_t) * self.bpm / 60
            wraps = int(math.floor(advanced))
            self.beat_index += wraps
            self.phase = advanced - wraps
        self.last_t = t
        self.envelope.append(max(0.0, float(strength)))
        self.steps += 1
        if len(self.envelope) >= MIN_HISTORY and self.steps % UPDATE_EVERY == 0:
            self._estimate()
        if kick is not None and self.bpm > 0 and min(self.phase, 1 - self.phase) < .15:
            position = (self.beat_index + (1 if self.phase > .5 else 0)) % 4
            self.kick_by_position *= .9
            self.kick_by_position[position] += kick.strength
        downbeat = int(np.argmax(self.kick_by_position)) if self.kick_by_position.any() else 0
        bar_phase = (((self.beat_index - downbeat) % 4) + self.phase) / 4 if self.bpm > 0 else 0.0
        return BeatState(self.bpm, self.phase, bar_phase % 1.0, self.confidence, t)

    def _estimate(self) -> None:
        env = np.convolve(np.asarray(self.envelope), SMOOTH, mode="same")
        centred = env - env.mean()
        energy = float(np.dot(centred, centred))
        if energy <= 1e-12:
            self.confidence = 0.0
            return
        lags = np.arange(int(60 / MAX_BPM / STEP) - 1, int(60 / MIN_BPM / STEP) + 2)
        ac = np.array([np.dot(centred[:-lag], centred[lag:]) for lag in lags]) / energy
        bpms = 60 / (lags * STEP)
        prior = np.exp(-.5 * (np.log2(bpms / PRIOR_BPM) / PRIOR_OCTAVES) ** 2)
        valid = (bpms >= MIN_BPM) & (bpms <= MAX_BPM)
        best = int(np.argmax(np.where(valid, ac * prior, -np.inf)))
        offset = 0.0
        if 0 < best < len(ac) - 1:
            a, b, c = ac[best - 1], ac[best], ac[best + 1]
            denominator = a - 2 * b + c
            if denominator < 0:
                offset = float(np.clip(.5 * (a - c) / denominator, -.5, .5))
        period = (lags[best] + offset)  # in envelope samples
        estimate = 60 / (period * STEP)
        self.estimates.append(estimate)
        stability = 1 - min(1.0, float(np.std(self.estimates)) / 3) if len(self.estimates) > 1 else .5
        self.confidence = float(np.clip((ac[best] - .1) / .4, 0, 1) * stability)
        self.bpm = float(np.median(self.estimates))
        period = 60 / (self.bpm * STEP)
        measured = self._comb_phase(env, period)
        if self.confidence > 0 and self.estimates and len(self.estimates) == 1:
            self.phase = measured
        else:
            error = (measured - self.phase + .5) % 1 - .5
            self.phase = (self.phase + PHASE_PULL * error) % 1.0

    @staticmethod
    def _comb_phase(env: np.ndarray, period: float) -> float:
        """Phase of 'now' (0 = on a beat) that best explains the last COMB_WINDOW samples."""
        now = len(env) - 1
        candidates = np.arange(0, period, .25)
        scores = []
        for since in candidates:
            total, j = 0.0, 0
            while since + j * period < min(COMB_WINDOW, now):
                total += _interp(env, now - since - j * period)
                j += 1
            scores.append(total / max(1, j))
        since = float(candidates[int(np.argmax(scores))])
        return (since / period) % 1.0
