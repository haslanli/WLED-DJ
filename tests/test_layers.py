import numpy as np

from signals import hats, noise_floor, sine
from wled_dj.analysis.features import Features
from wled_dj.analysis.frontend import HOP, FrontEnd
from wled_dj.analysis.layers import LAYERS, Layers, LayerTracker, bands_power


def P(db):
    """Mean power of a signal at `db` relative to a full-scale sine."""
    return .5 * 10 ** (db / 10)


QUIET = P(-120)


def run(tracker, seconds, drums=QUIET, bass=QUIET, vocals=QUIET, other=QUIET, mix=None, kick=QUIET):
    """Step the tracker with constant powers; returns the Layers after every hop."""
    out = []
    for _ in range(int(round(seconds * 100))):
        power = {"drums": drums, "bass": bass, "vocals": vocals, "other": other}
        out.append(tracker.step(power, sum(power.values()) if mix is None else mix, kick))
    return out


def test_level_reads_db():
    last = run(LayerTracker("stems"), 1.0, drums=P(-20), mix=P(-18))[-1]
    assert isinstance(last, Layers) and last.source == "stems"
    assert abs(last.drums.level_db + 20) < .5 and abs(last.drums.fast_db + 20) < .5
    assert abs(last.drums.rel_db + 2) < .5
    assert last["drums"] is last.drums and set(LAYERS) == {"drums", "bass", "vocals", "other"}


def test_enters_after_150_ms():
    tracker = LayerTracker("stems")
    run(tracker, 1.0, mix=P(-70))
    after = run(tracker, .3, drums=P(-20), mix=P(-18))
    assert not after[9].drums.present            # 100 ms in
    assert after[24].drums.present               # 250 ms in


def test_leaves_after_a_second():
    tracker = LayerTracker("stems")
    playing = run(tracker, 2.0, drums=P(-20), mix=P(-18))
    assert playing[-1].drums.present and playing[-1].drums.since_s > 1.5
    gone = run(tracker, 2.0, drums=P(-90), mix=P(-18))
    assert gone[89].drums.present                # 0.9 s after it stopped
    assert not gone[159].drums.present           # 1.6 s after
    assert gone[-1].drums.since_s < .7           # the clock restarted when it left


def test_gaps_between_kicks_are_not_exits():
    tracker = LayerTracker("stems")
    seen = []
    for hop in range(500):
        level = P(-10) if hop % 50 < 2 else QUIET
        seen.append(tracker.step({"drums": level, "bass": QUIET, "vocals": QUIET, "other": QUIET}, level, QUIET))
    assert all(layers.drums.present for layers in seen[30:])


def test_far_under_the_mix_is_absent():
    seen = run(LayerTracker("stems"), 3.0, vocals=P(-48), mix=P(-18))
    assert not any(layers.vocals.present for layers in seen)


def test_silence_has_no_layers():
    seen = run(LayerTracker("stems"), 3.0, drums=P(-72), bass=P(-72), vocals=P(-72), other=P(-72))
    assert not any(layers[name].present for layers in seen for name in LAYERS)


def test_kick_weight():
    last = run(LayerTracker("stems"), 2.0, drums=P(-12), kick=P(-12))[-1]
    assert abs(last.kick_weight_db + 12) < 1


def total_power(signal):
    front = FrontEnd()
    totals = {name: 0.0 for name in LAYERS}
    for i in range(len(signal) // HOP):
        power, mix, kick = bands_power(front.push(signal[i * HOP:(i + 1) * HOP]))
        assert all(value >= 0 for value in power.values()) and mix >= 0 and kick >= 0
        for name in LAYERS:
            totals[name] += power[name]
    return totals


def test_bands_power_separates_tone_from_noise():
    tone = total_power(sine(100, 2.0, .5))
    assert tone["bass"] > 10 * max(tone["drums"], tone["vocals"], tone["other"])
    bursts = total_power(noise_floor(2.0, -70) + hats(np.arange(.2, 1.9, .1), 2.0))
    assert bursts["drums"] > max(bursts["bass"], bursts["vocals"], bursts["other"])


def test_features_carry_layers():
    f = Features(t=0.0)
    assert f.layers is None and f.source == "bands"


# ---- stems -> events and layers ----------------------------------------------------------------

from signals import kick, stab  # noqa: E402
from wled_dj.analysis.reading import LayerAnalyzer, Reading  # noqa: E402
from wled_dj.stems.link import StemHop  # noqa: E402


def mono(signal):
    return signal.mean(axis=1).astype(np.float32)


def stem_hops(seconds, drums=None, bass=None, vocals=None, other=None):
    """StemHops at 44.1 kHz from whole mono signals (missing stems are silent)."""
    n = int(seconds * 100)
    silent = np.zeros(n * 441, np.float32)
    stems = {"drums": drums, "bass": bass, "vocals": vocals, "other": other}
    stems = {name: silent if value is None else value[:n * 441] for name, value in stems.items()}
    for i in range(n):
        part = {name: value[i * 441:(i + 1) * 441] for name, value in stems.items()}
        yield StemHop(i, (i + 1) * .01, part["drums"], part["bass"], part["vocals"], part["other"],
                      part["drums"] + part["bass"] + part["vocals"] + part["other"])


def analyse(hops):
    analyzer = LayerAnalyzer()
    events = {name: [] for name in ("kick", "snare", "hats", "bass", "vocal", "stab")}
    layers = []
    for hop in hops:
        reading = analyzer.step(hop)
        assert isinstance(reading, Reading) and set(reading.events) == set(events)
        found, state = reading.events, reading.layers
        for name, onset in found.items():
            if onset is not None:
                assert onset.name == name and onset.pan == 0.0
                events[name].append(hop.t)
        layers.append(state)
    return events, layers


def test_kick_in_the_drums_stem_is_a_kick():
    starts = [2.0, 2.5, 3.0, 3.5]
    events, _ = analyse(stem_hops(4.5, drums=mono(kick(starts, 4.5, rate=44100)), bass=mono(sine(55, 4.5, .5, rate=44100))))
    assert len(events["kick"]) == 4 and events["stab"] == []
    assert all(0 <= t - s <= .025 for t, s in zip(events["kick"], starts))


def test_bass_tone_in_the_bass_stem_is_no_kick():
    events, layers = analyse(stem_hops(3.0, bass=mono(sine(55, 3.0, .5, rate=44100))))
    assert events["kick"] == events["snare"] == events["hats"] == []
    assert layers[-1].bass.present and not layers[-1].drums.present


def test_hats_and_presence():
    times = np.arange(1.0, 2.9, .125)
    events, layers = analyse(stem_hops(3.0, drums=mono(hats(times, 3.0, rate=44100))))
    assert len(events["hats"]) >= len(times) - 2 and events["kick"] == []
    assert layers[130].drums.present and not layers[90].drums.present
    assert layers[-1].source == "stems"


def test_a_chord_in_the_other_stem_is_a_stab():
    starts = [2.0, 2.5, 3.0, 3.5]
    floor = mono(noise_floor(4.5, -70, rate=44100))
    events, layers = analyse(stem_hops(4.5, other=floor + mono(stab(starts, 4.5, rate=44100))))
    assert len(events["stab"]) == 4
    assert all(0 <= t - s <= .05 for t, s in zip(events["stab"], starts))
    assert events["kick"] == events["snare"] == events["hats"] == events["vocal"] == []
    assert layers[-1].other.present


def test_a_chord_in_the_vocals_stem_is_not_a_stab():
    floor = mono(noise_floor(4.5, -70, rate=44100))
    events, _ = analyse(stem_hops(4.5, vocals=floor + mono(stab([2.0, 2.5, 3.0, 3.5], 4.5, rate=44100))))
    assert events["stab"] == [] and len(events["vocal"]) == 4


def test_a_held_chord_is_one_stab_at_most():
    held = sum(mono(sine(f, 3.0, .1, rate=44100)) for f in (440.0, 554.37, 659.25))
    events, _ = analyse(stem_hops(3.0, other=held))
    assert [t for t in events["stab"] if t > .3] == []
