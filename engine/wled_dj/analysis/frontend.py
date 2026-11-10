"""Causal spectral front end: 1024-point Hann STFT every 10 ms (480 samples at 48 kHz by default),
with a causal harmonic/percussive split and per-channel magnitudes for stereo panning.
It also runs at other rates and on mono signals, which is how separated stems are read."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfilt

RATE = 48000
HOP = 480
HOP_S = .01              # every front end steps 10 ms, whatever its rate
NFFT = 1024
FREQ = np.fft.rfftfreq(NFFT, 1 / RATE)
WINDOW = np.hanning(NFFT).astype(np.float32)
HARMONIC_FRAMES = 9      # current + 8 previous frames, median per bin
PERCUSSIVE_BINS = 17     # median across frequency in the current frame
# One-sided Hann-windowed energy of a full-scale sine: 3 * NFFT^2 / 32 (Parseval). Used for dBFS levels.
FULL_SCALE_ENERGY = 3 * NFFT * NFFT / 32
# Kicks live below what a 1024-point FFT can resolve; a band filter gives a stable low envelope.
LOW_BAND = (35.0, 130.0)
LOW_SOS = butter(4, LOW_BAND, "bandpass", fs=RATE, output="sos")


@dataclass(frozen=True)
class Spectrum:
    mag: np.ndarray
    perc: np.ndarray
    harm: np.ndarray
    mag_left: np.ndarray
    mag_right: np.ndarray
    low_power: float = 0.0     # mean power of the 35-130 Hz band over this hop (full-scale sine = 0.5)
    low_left: float = 0.0
    low_right: float = 0.0


def band(lo: float, hi: float, rate: int = RATE) -> np.ndarray:
    freq = FREQ if rate == RATE else np.fft.rfftfreq(NFFT, 1 / rate)
    return (freq >= lo) & (freq < hi)


def level_dbfs(mag: np.ndarray, mask: np.ndarray) -> float:
    energy = float(np.sum(mag[mask] ** 2))
    return 10 * np.log10(energy / FULL_SCALE_ENERGY + 1e-20)


class FrontEnd:
    def __init__(self, rate: int = RATE, hop: int = HOP, channels: int = 2):
        self.rate, self.hop, self.channels = rate, hop, channels
        self.shape = (hop, 2) if channels == 2 else (hop,)
        self.buffer = np.zeros((NFFT, channels), np.float32)
        self.history: deque[np.ndarray] = deque(maxlen=HARMONIC_FRAMES - 1)
        self.low_sos = LOW_SOS if rate == RATE else butter(4, LOW_BAND, "bandpass", fs=rate, output="sos")
        self.low_state = np.zeros((self.low_sos.shape[0], 2, channels))

    band = staticmethod(band)

    def push(self, hop: np.ndarray) -> Spectrum:
        if hop.shape != self.shape:
            raise ValueError(f"expected {self.shape} samples, got {hop.shape}")
        hop = hop.reshape(self.hop, self.channels)
        self.buffer[:-self.hop] = self.buffer[self.hop:]
        self.buffer[-self.hop:] = hop
        spectra = np.abs(np.fft.rfft(self.buffer * WINDOW[:, None], axis=0)).astype(np.float32)
        if self.channels == 2:
            left, right = spectra[:, 0], spectra[:, 1]
            mag = np.abs(np.fft.rfft(self.buffer.mean(axis=1) * WINDOW)).astype(np.float32)
        else:
            mag = left = right = spectra[:, 0]
        stack = np.vstack((*self.history, mag)) if self.history else mag[None]
        harmonic = np.median(stack, axis=0)
        percussive = median_filter(mag, size=PERCUSSIVE_BINS, mode="nearest")
        self.history.append(mag)
        mask = percussive ** 2 / (percussive ** 2 + harmonic ** 2 + 1e-12)
        low, self.low_state = sosfilt(self.low_sos, hop, axis=0, zi=self.low_state)
        low_left, low_right = np.broadcast_to((low ** 2).mean(axis=0), (2,))
        return Spectrum(mag, mag * mask, mag * (1 - mask), left, right,
                        float((low_left + low_right) / 2), float(low_left), float(low_right))
