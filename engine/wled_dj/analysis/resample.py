"""Streaming 48 kHz -> 44.1 kHz (the rate the stem model works at): 160 samples in, 147 out.

One causal low-pass at the common multiple of both rates. Its passband reaches 18 kHz; whatever the 48 kHz
audio holds above 22 kHz can only fold to above 20 kHz, where no detector reads. Delay: a quarter millisecond."""
from __future__ import annotations

import numpy as np
from scipy.signal import firwin, upfirdn

UP, DOWN = 147, 160
TAPS_PER_PHASE = 24
SOURCE_RATE = 48000


class Resampler:
    def __init__(self):
        taps = UP * TAPS_PER_PHASE + 1
        self.h = firwin(taps, 22000, fs=SOURCE_RATE * UP, window=("kaiser", 5.65)) * UP
        self.delay_s = (taps - 1) / 2 / (SOURCE_RATE * UP)
        self.tail: np.ndarray | None = None        # the previous DOWN input samples

    def push(self, x: np.ndarray) -> np.ndarray:
        """x: (160 * k, channels) at 48 kHz -> (147 * k, channels) float32 at 44.1 kHz."""
        if len(x) == 0 or len(x) % DOWN:
            raise ValueError(f"expected a multiple of {DOWN} samples, got {len(x)}")
        if self.tail is None:
            self.tail = np.zeros((DOWN,) + x.shape[1:], np.float64)
        extended = np.concatenate((self.tail, x))
        self.tail = extended[-DOWN:]
        # DOWN samples of history shift the output by exactly UP samples, so the new block's part starts at UP.
        out = upfirdn(self.h, extended, UP, DOWN, axis=0)
        return out[UP:UP + len(x) // DOWN * UP].astype(np.float32)
