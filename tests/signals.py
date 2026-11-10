"""Synthetic stereo test signals at 48 kHz, float32 (n, 2)."""
import numpy as np
from scipy.signal import butter, sosfilt

RATE = 48000


def _empty(dur, rate=RATE):
    return np.zeros((int(round(dur * rate)), 2), np.float32)


def noise_floor(dur, db=-70, seed=0, rate=RATE):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((int(round(dur * rate)), 2)) * 10 ** (db / 20)).astype(np.float32)


def sine(freq, dur, amp=.5, rate=RATE):
    t = np.arange(int(round(dur * rate))) / rate
    wave = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.column_stack((wave, wave))


def kick(t0_list, dur, amp=.5, freq=60.0, rate=RATE):
    """Decaying low sine bursts (120 ms long, 40 ms time constant) starting at each t0."""
    out = _empty(dur, rate)
    n = int(.12 * rate)
    t = np.arange(n) / rate
    burst = amp * np.sin(2 * np.pi * freq * t) * np.exp(-t / .04)
    for t0 in t0_list:
        start = int(round(t0 * rate))
        part = burst[:max(0, min(n, len(out) - start))]
        out[start:start + len(part)] += part[:, None]
    return out


def hats(t0_list, dur, amp=.3, channel=None, seed=1, rate=RATE):
    """30 ms high-passed (>7 kHz) noise bursts; channel 0/1 restricts to left/right."""
    out = _empty(dur, rate)
    rng = np.random.default_rng(seed)
    sos = butter(4, 7000, 'highpass', fs=rate, output='sos')
    n = int(.03 * rate)
    env = np.exp(-np.arange(n) / rate / .01)
    for t0 in t0_list:
        burst = sosfilt(sos, rng.standard_normal(n)) * env * amp
        start = int(round(t0 * rate))
        part = burst[:max(0, min(n, len(out) - start))]
        if channel is None:
            out[start:start + len(part)] += part[:, None]
        else:
            out[start:start + len(part), channel] += part
    return out


def music_pattern(dur, bpm=128.0, start=0.0, rate=RATE):
    """Kick on every beat, hats on the off-beats, over a quiet floor."""
    beat = 60.0 / bpm
    kicks = np.arange(start, dur, beat)
    offs = kicks + beat / 2
    return noise_floor(dur, -70, rate=rate) + kick(kicks, dur, rate=rate) + hats(offs[offs < dur], dur, rate=rate)


def stab(t0_list, dur, amp=.3, freqs=(440.0, 554.37, 659.25), rate=RATE):
    """Chord stabs: three sines struck together (200 ms long, 80 ms time constant) starting at each t0."""
    out = _empty(dur, rate)
    n = int(.2 * rate)
    t = np.arange(n) / rate
    burst = amp / len(freqs) * sum(np.sin(2 * np.pi * f * t) for f in freqs) * np.exp(-t / .08)
    for t0 in t0_list:
        start = int(round(t0 * rate))
        part = burst[:max(0, min(n, len(out) - start))]
        out[start:start + len(part)] += part[:, None]
    return out
