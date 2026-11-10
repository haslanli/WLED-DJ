"""The lead voice: tones in the vocals and other stems in, where the lead sits and how it moves out."""
import math

import numpy as np

from wled_dj.analysis.frontend import FrontEnd
from wled_dj.analysis.layers import LayerTracker
from wled_dj.analysis.lead import LeadTracker, Voice

RATE, HOP = 44100, 441
QUIET = .5e-12


def tone(hz, seconds, amp=.2):
    """A sine; hz may be a function of time in seconds, for a glide."""
    t = np.arange(int(seconds * RATE)) / RATE
    f = np.vectorize(hz)(t) if callable(hz) else np.full(len(t), float(hz))
    return (amp * np.sin(2 * np.pi * np.cumsum(f) / RATE)).astype(np.float32)


class Stage:
    """Feeds the vocals and other stems hop by hop through the real front end and layer tracker."""

    def __init__(self):
        self.fronts = {name: FrontEnd(RATE, HOP, 1) for name in ("vocals", "other")}
        self.layers, self.lead = LayerTracker("stems"), LeadTracker(RATE)

    def play(self, seconds, other=None, vocals=None):
        silent = np.zeros(int(seconds * RATE), np.float32)
        stems = {"vocals": silent if vocals is None else vocals, "other": silent if other is None else other}
        out = []
        for i in range(int(round(seconds * 100))):
            part = {name: x[i * HOP:(i + 1) * HOP] for name, x in stems.items()}
            spectra = {name: self.fronts[name].push(part[name]) for name in part}
            power = {name: float(np.mean(part[name].astype(np.float64) ** 2)) + QUIET for name in part}
            layers = self.layers.step({"drums": QUIET, "bass": QUIET, **power}, sum(power.values()), QUIET)
            out.append(self.lead.step(spectra["vocals"], spectra["other"], layers))
        return out


def glide(t):
    return 770 * (450 / 770) ** (t / 9)


def test_a_falling_lead_is_followed():
    stage = Stage()
    for hz in (400, 800, 400, 800):                  # the lead has been low and high before: that is its range
        stage.play(1.0, other=tone(hz, 1.0))
    stage.play(1.0, other=tone(770, 1.0))
    heard = stage.play(9.0, other=tone(glide, 9.0))
    for second in (2, 5, 8):
        assert abs(heard[second * 100].hz / glide(second) - 1) < .1
    assert isinstance(heard[0], Voice) and heard[100].position > .8 and heard[-1].position < .2
    falling = [v.velocity < 0 for v in heard[100:]]
    assert sum(falling) >= .9 * len(falling)
    assert all(0 <= v.position <= 1 and not v.vocal and v.level > .5 for v in heard)


def test_a_steady_lead_does_not_move():
    heard = Stage().play(4.0, other=tone(600, 4.0))
    assert all(abs(v.velocity) < .02 and v.position == .5 for v in heard[100:])
    assert abs(heard[-1].hz / 600 - 1) < .1


def test_a_voice_takes_the_lead():
    sung = Stage().play(2.0, other=tone(2000, 2.0), vocals=tone(300, 2.0))[-1]
    assert sung.vocal and abs(sung.hz / 300 - 1) < .1
    played = Stage().play(2.0, other=tone(2000, 2.0))[-1]
    assert not played.vocal and abs(played.hz / 2000 - 1) < .1


def test_no_lead_is_no_motion():
    stage = Stage()
    assert stage.play(1.0)[-1] == Voice()
    stage.play(2.0, other=tone(600, 2.0))
    after = stage.play(2.0)
    assert all(v.level == 0 and v.velocity == 0 for v in after[10:])
    assert abs(after[-1].hz / 600 - 1) < .1 and after[-1].position == after[10].position      # it holds where it was
    assert all(math.isfinite(x) for v in after for x in (v.position, v.velocity, v.level, v.hz))


def test_travel_is_the_way_made_not_the_wobble():
    falling = Stage()
    for hz in (400, 800, 400, 800):
        falling.play(1.0, other=tone(hz, 1.0))
    falling.play(1.0, other=tone(770, 1.0))
    falling.play(9.0, other=tone(glide, 9.0))
    assert falling.lead.travel > .15                           # it went somewhere in the last four seconds
    wobbling = Stage()
    for hz in (400, 800, 400, 800):
        wobbling.play(1.0, other=tone(hz, 1.0))
    wobbling.play(8.0, other=tone(lambda t: 600 * 2 ** (math.sin(2 * math.pi * 5 * t) / 12), 8.0))     # a semitone of vibrato
    assert wobbling.lead.travel < .05
    assert Stage().lead.travel == 0
