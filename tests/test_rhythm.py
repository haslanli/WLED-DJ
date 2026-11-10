import numpy as np

from wled_dj.analysis.onsets import Onset
from wled_dj.analysis.rhythm import BeatTracker

STEP = .01


def clicks(bpm, start, end):
    return np.arange(start, end, 60 / bpm)


def feed(click_times, duration, noise=0.0, seed=0):
    """Feed a strength envelope with 1.0 at each click (and optional noise); return (times, states)."""
    rng = np.random.default_rng(seed)
    tracker = BeatTracker()
    hops = set(int(round(c / STEP)) for c in click_times)
    times, states = [], []
    for i in range(int(round(duration / STEP))):
        t = i * STEP
        hit = i in hops
        strength = (1.0 if hit else 0.0) + noise * rng.random()
        states.append(tracker.step(t, strength, Onset("kick", 1.0, 0.0) if hit else None))
        times.append(t)
    return np.array(times), states


def beat_times(times, states, after):
    """Times where the reported phase wraps around (a predicted beat)."""
    out = []
    for i in range(1, len(states)):
        if times[i] >= after and states[i].bpm > 0 and states[i].phase < states[i - 1].phase - .5:
            out.append(times[i] - states[i].phase * 60 / states[i].bpm)
    return np.array(out)


def test_click_track_128bpm():
    true = clicks(128, .3, 10)
    times, states = feed(true, 10)
    state = states[-1]
    assert abs(state.bpm - 128) <= 1.5 and state.confidence > .6
    predicted = beat_times(times, states, after=6)
    assert len(predicted) >= 7
    errors = [np.min(np.abs(true - p)) for p in predicted]
    assert max(errors) <= .025


def test_tempo_change_tracked():
    true = np.concatenate((clicks(120, .3, 10), clicks(100, 10, 18.5)))
    times, states = feed(true, 18.5)
    after = [s for t, s in zip(times, states) if t >= 18.0]
    assert all(abs(s.bpm - 100) <= 2 for s in after)


def test_noise_has_low_confidence():
    rng = np.random.default_rng(3)
    tracker = BeatTracker()
    for i in range(1000):
        state = tracker.step(i * STEP, float(rng.random()), None)
    assert state.confidence < .3


def test_phase_at_predicts_forward():
    _, states = feed(clicks(128, .3, 8), 8)
    s = states[-1]
    assert s.bpm > 0
    expected = (s.phase + .1 * s.bpm / 60) % 1
    assert abs(((s.phase_at(s.t + .1) - expected + .5) % 1) - .5) < .01


def test_unknown_tempo_before_enough_history():
    _, states = feed(clicks(128, .3, 1.5), 1.5)
    assert states[-1].bpm == 0 and states[-1].confidence == 0
    assert 0 <= states[-1].bar_phase < 1
