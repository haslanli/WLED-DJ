"""Cached stems replayed through the real analyzer (bench/replay.py), on made-up arrays."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

from replay import ReplayLink, cached_stems  # noqa: E402
from signals import RATE, music_pattern, sine  # noqa: E402
from wled_dj.analysis.analyzer import Analyzer  # noqa: E402
from wled_dj.capture import AudioBlock  # noqa: E402
from wled_dj.stems.link import StemLink  # noqa: E402

QUIET48 = np.zeros((480, 2), np.float32)


def test_replayed_stems_are_aligned():
    n = 441 * 6
    est = np.zeros((n, 3), np.float32)
    est[:, 0] = np.arange(n)                                   # drums: every sample says where it is
    link = ReplayLink(est, np.zeros(n, np.float32))
    assert link.state == "on" and link.latency_s == StemLink(["true"]).latency_s
    assert [link.push(QUIET48, 10.0 + i * .01) for i in range(5)] == [0, 1, 2, 3, 4]
    hops = link.collect(0.0)
    assert [hop.index for hop in hops] == [0, 1, 2, 3, 4] and link.collect(0.0) == []
    assert np.array_equal(hops[3].drums, np.arange(1067, 1508, dtype=np.float32))      # 441 * 3 - 256 on
    assert np.array_equal(hops[0].drums, np.concatenate((np.zeros(256, np.float32), np.arange(185, dtype=np.float32))))
    assert hops[3].drums.shape == hops[3].other.shape == hops[3].mix.shape == (441,)
    assert hops[3].t == pytest.approx(10.03 - link.latency_s)


def test_other_is_the_exact_remainder():
    n = 441 * 4
    rng = np.random.default_rng(5)
    est = (rng.standard_normal((n, 3)) * .1).astype(np.float32)
    tone = sine(1000, n / 44100, .2, rate=44100)[:n, 0]
    link = ReplayLink(est, est.sum(axis=1) + tone)
    for i in range(4):
        link.push(QUIET48, i * .01)
    padded = np.concatenate((np.zeros(256, np.float32), tone))
    for hop in link.collect(0.0):
        assert np.allclose(hop.other, padded[441 * hop.index:441 * (hop.index + 1)], atol=1e-6)
    assert cached_stems(Path("no-such-clip.wav")) is None


def test_the_analyzer_reads_replayed_stems():
    audio = music_pattern(3.0)
    mono44 = music_pattern(3.0, rate=44100).mean(axis=1).astype(np.float32)
    est = np.stack((mono44, np.zeros_like(mono44), np.zeros_like(mono44)), axis=1)      # all of it is drums
    analyzer = Analyzer(ReplayLink(est, mono44))
    fs = []
    for start in range(0, len(audio) - 511, 512):
        fs.extend(analyzer.feed(AudioBlock(audio[start:start + 512], start / RATE, 0), 0.0))
    assert all(f.source == "stems" for f in fs[100:]) and fs[150].layers.drums.present
    kicks = [f.t for f in fs if f.kick is not None and f.t >= 1.0]
    expected = len([k for k in np.arange(0, 3.0, 60 / 128) if 1.0 <= k < 2.95])
    assert abs(len(kicks) - expected) <= 1 and min(np.diff(kicks)) > .3


@pytest.mark.models
def test_a_new_file_is_separated_and_cached(tmp_path, monkeypatch):
    """Music added to bench/corpus has no cached stems: the report separates it with the real model first."""
    from scipy.io import wavfile

    import replay
    import structure_report
    from wled_dj.stems.models import models_dir, verified
    if not verified(models_dir()):
        pytest.skip(f"no verified model files in {models_dir()}")
    song = tmp_path / "a new song.wav"
    wavfile.write(song, RATE, (music_pattern(4.0) * 32767).astype(np.int16))
    monkeypatch.setattr(replay, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(structure_report, "CACHE", tmp_path / "cache")
    assert cached_stems(song) is None
    structure_report.separate(song)
    est = cached_stems(song)
    assert est.shape[1] == 3 and abs(len(est) - 4 * 44100) < 300
    assert np.mean(est[44100:, 0] ** 2) > 10 * np.mean(est[44100:, 2] ** 2)         # kicks and hats are drums, not a voice
    features = replay.analyse(song)
    assert features[-1].source == "stems" and features[-1].layers.drums.present
