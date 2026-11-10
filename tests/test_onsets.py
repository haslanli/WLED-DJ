import numpy as np

from signals import RATE, hats, kick, noise_floor, sine
from wled_dj.analysis.frontend import HOP, FrontEnd
from wled_dj.analysis.onsets import SPECS, OnsetDetector


def run(signal):
    """Feed a (n, 2) signal hop by hop; return {detector name: [(time, onset), ...]}."""
    front = FrontEnd()
    detectors = [OnsetDetector(spec) for spec in SPECS]
    found = {spec.name: [] for spec in SPECS}
    for i in range(len(signal) // HOP):
        spectrum = front.push(signal[i * HOP:(i + 1) * HOP])
        t = (i + 1) * HOP / RATE
        for detector in detectors:
            onset = detector.step(spectrum)
            if onset is not None:
                found[onset.name].append((t, onset))
    return found


def test_kick_burst_detected_promptly():
    starts = [2.0 + .5 * i for i in range(6)]
    found = run(noise_floor(5.2, -70) + kick(starts, 5.2))
    times = [t for t, _ in found["kick"]]
    assert len(times) == 6
    delays = [min(t - s for t in times if t >= s) for s in starts]
    assert all(0 <= d <= .025 for d in delays)
    assert all(0 < o.strength <= 1 for _, o in found["kick"])


def test_sustained_bass_is_not_a_kick():
    found = run(sine(55, 3.0, .5))
    assert [t for t, _ in found["kick"] if t > .2] == []


def test_sustained_bass_is_not_repeated_bass_notes():
    found = run(sine(55, 3.0, .5))
    assert [t for t, _ in found["bass"] if t > .3] == []


def test_kick_over_sustained_bass_still_detected():
    starts = [1.0 + .5 * i for i in range(4)]
    found = run(sine(45, 3.0, .3) + kick(starts, 3.0, freq=70))
    assert len([t for t, _ in found["kick"] if t > .3]) == 4


def test_hats_from_noise_bursts():
    starts = [.25 * i for i in range(1, 13)]
    found = run(noise_floor(3.25, -70) + hats(starts, 3.25))
    assert abs(len(found["hats"]) - 12) <= 1
    assert found["kick"] == []


def test_refractory_limits_rate():
    starts = list(np.arange(.5, 2.5, .05))
    times = [t for t, _ in run(kick(starts, 2.6))["kick"]]
    assert len(times) >= 2 and min(np.diff(times)) >= .09 - 1e-9


def test_silence_gives_nothing():
    found = run(np.zeros((3 * RATE, 2), np.float32))
    assert all(v == [] for v in found.values())


def test_pan_follows_channel():
    starts = [.25 * i for i in range(1, 9)]
    found = run(hats(starts, 2.25, channel=1))
    assert found["hats"] and all(o.pan > .8 for _, o in found["hats"])


def run_mono(signal, rate=44100, hop=441):
    """The stems path: a mono signal at 44.1 kHz. Returns {name: [time, ...]}."""
    front = FrontEnd(rate, hop, 1)
    detectors = [OnsetDetector(spec, rate) for spec in SPECS]
    found = {spec.name: [] for spec in SPECS}
    for i in range(len(signal) // hop):
        spectrum = front.push(signal[i * hop:(i + 1) * hop])
        for detector in detectors:
            if detector.step(spectrum) is not None:
                found[detector.spec.name].append((i + 1) * .01)
    return found


def test_detectors_work_at_44100_mono():
    kicks, hat_times = [2.0, 2.5, 3.0], [2.25, 2.75]
    stereo = noise_floor(4, -70, rate=44100) + kick(kicks, 4, rate=44100) + hats(hat_times, 4, rate=44100)
    found = run_mono(stereo.mean(axis=1).astype(np.float32))
    assert len(found["kick"]) == 3 and len(found["hats"]) == 2
    assert all(0 <= t - s <= .025 for t, s in zip(found["kick"], kicks))
    assert all(0 <= t - s <= .025 for t, s in zip(found["hats"], hat_times))


def test_mono_front_end_shapes():
    front = FrontEnd(44100, 441, 1)
    spectrum = front.push(np.zeros(441, np.float32))
    assert spectrum.mag.shape == spectrum.perc.shape == spectrum.harm.shape == (513,)
    assert spectrum.mag_left.shape == spectrum.mag_right.shape == (513,)
    try:
        front.push(np.zeros((441, 2), np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("a stereo hop must be refused by a mono front end")


def test_pan_is_kept_every_hop():
    signal = noise_floor(3, -70) + hats([2.0], 3, channel=1)
    front = FrontEnd()
    detector = OnsetDetector(next(spec for spec in SPECS if spec.name == "hats"))
    assert detector.pan == 0.0
    pans = []
    for i in range(len(signal) // HOP):
        if detector.step(front.push(signal[i * HOP:(i + 1) * HOP])) is not None:
            pans.append(detector.pan)
    assert len(pans) == 1 and pans[0] > .8
