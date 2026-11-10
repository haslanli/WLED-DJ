"""Cached stems through the real analyzer, offline: the engine's own analysis on a whole file, with no model
and no clock.

A ReplayLink stands where the engine's StemLink does. The analyzer pushes audio into it and gets that audio's
stems back at once, as from a helper that is never late. The stems come from bench/data/stems-cache/corpus
(StemgenRT's output for the files in bench/corpus, see bench/DATA.md).

    from replay import analyse
    features = analyse(Path("bench/corpus/song.wav"))      # one Features per 10 ms
    drums = analyse(Path("bench/corpus/song.wav"), layer="drums")      # as if only the drums played"""
from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
from scipy.signal import resample_poly

from wled_dj.analysis.analyzer import Analyzer
from wled_dj.analysis.features import Features
from wled_dj.analysis.frontend import RATE
from wled_dj.analysis.resample import Resampler
from wled_dj.capture import AudioBlock
from wled_dj.stems import protocol
from wled_dj.stems.link import STEM_HOP, StemHop

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "stems-cache" / "corpus"
BLOCK = 512                       # samples per block fed to the analyzer, as a real capture sends them
LAYERS = ("drums", "bass", "vocals", "other")


def decode(path: Path, rate: int) -> np.ndarray:
    """The whole file as (n, 2) float32 at `rate`."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "f32le", "-ac", "2", "-ar", str(rate), "-"]
    return np.frombuffer(subprocess.check_output(cmd), "<f4").reshape(-1, 2).copy()


def cached_stems(audio: Path) -> np.ndarray | None:
    """(n, 3) float32 at 44.1 kHz (drums, bass, vocals) for a file in bench/corpus, or None when not cached."""
    target = CACHE / (Path(audio).stem + ".npz")
    if not target.exists():
        return None
    return np.load(target)["est"].astype(np.float32) / 32767


class ReplayLink:
    """est: (n, 3) stems; mix: (n,) the mono mix they were made from; both at 44.1 kHz, sample for sample.
    Stem hop n is samples [441 n - 256, 441 (n + 1) - 256), the stretch a StemLink would hand back."""
    state = "on"

    def __init__(self, est: np.ndarray, mix: np.ndarray):
        self.latency_s = Resampler().delay_s + protocol.LATENCY / protocol.RATE
        lead = protocol.LATENCY
        self._est = np.concatenate((np.zeros((lead, 3), np.float32), np.asarray(est, np.float32)))
        self._mix = np.concatenate((np.zeros(lead, np.float32), np.asarray(mix, np.float32)))
        self._pushed: list[tuple[int, float]] = []
        self._count = 0

    def push(self, hop48: np.ndarray, t_end: float) -> int:
        index = self._count
        self._count += 1
        self._pushed.append((index, t_end))
        return index

    def collect(self, wait_s: float) -> list[StemHop]:
        pushed, self._pushed = self._pushed, []
        return [self._hop(index, t_end) for index, t_end in pushed]

    def _hop(self, index: int, t_end: float) -> StemHop:
        start = index * STEM_HOP
        est, mix = _stretch(self._est, start), _stretch(self._mix, start)
        drums, bass, vocals = est[:, 0], est[:, 1], est[:, 2]
        return StemHop(index, t_end - self.latency_s, drums, bass, vocals, mix - drums - bass - vocals, mix)


def _stretch(samples: np.ndarray, start: int) -> np.ndarray:
    """STEM_HOP samples from `start`, zeros past the end."""
    part = samples[start:start + STEM_HOP]
    if len(part) == STEM_HOP:
        return part
    return np.concatenate((part, np.zeros((STEM_HOP - len(part),) + samples.shape[1:], samples.dtype)))


def _alone(audio: Path, layer: str) -> tuple[np.ndarray, np.ndarray]:
    """(stems, mix) at 44.1 kHz of the file as if only this layer played: drums, bass, vocals or other."""
    est = cached_stems(audio)
    if est is None:
        raise FileNotFoundError(f"no cached stems for {audio.name} in {CACHE}")
    mix = decode(audio, protocol.RATE).mean(axis=1)
    n = min(len(est), len(mix))
    alone = np.zeros((n, 3), np.float32)
    if layer == "other":                                  # what the three models leave over
        return alone, (mix[:n] - est[:n].sum(axis=1)).astype(np.float32)
    column = LAYERS.index(layer)
    alone[:, column] = est[:n, column]
    return alone, alone[:, column].copy()


def solo(audio: Path, layer: str) -> np.ndarray:
    """One layer of the file as sound: (n, 2) float32 at 48 kHz."""
    sound = resample_poly(_alone(Path(audio), layer)[1], 160, 147).astype(np.float32)
    return np.column_stack((sound, sound))


def analyse(audio: Path, stems: bool = True, seconds: float | None = None, layer: str | None = None) -> list[Features]:
    """The real Analyzer over the file (or its first `seconds`): on its cached stems, or on the mix alone.
    With `layer`, it hears that layer alone, as its sound and as its only stem. Host time is 0 at the file's
    first sample."""
    audio = Path(audio)
    link = None
    if layer is not None:
        link = ReplayLink(*_alone(audio, layer))
        samples = solo(audio, layer)
    elif stems:
        est = cached_stems(audio)
        if est is None:
            raise FileNotFoundError(f"no cached stems for {audio.name} in {CACHE}")
        mix = decode(audio, protocol.RATE).mean(axis=1)
        n = min(len(est), len(mix))
        link = ReplayLink(est[:n], mix[:n])
    if layer is None:
        samples = decode(audio, RATE)
    if seconds is not None:
        samples = samples[:int(seconds * RATE)]
    analyzer = Analyzer(link)
    features: list[Features] = []
    for start in range(0, len(samples) - BLOCK + 1, BLOCK):
        features.extend(analyzer.feed(AudioBlock(samples[start:start + BLOCK], start / RATE, 0), 0.0))
    return features
