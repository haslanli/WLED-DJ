"""Structure, driven by a real layer tracker: levels in dB go in, the song's state comes out."""
import numpy as np
import pytest

from wled_dj.analysis.layers import LayerTracker
from wled_dj.analysis.onsets import Onset
from wled_dj.analysis.structure import SECTIONS, Structure, StructureTracker

CLASSES = ("kick", "snare", "hats", "bass", "vocal", "stab")


def P(db):
    """Mean power of a signal at `db` relative to a full-scale sine; None is silence."""
    return .5 * 10 ** ((-120.0 if db is None else db) / 10)


class Band:
    """A band to conduct: play(seconds, ...) with each layer's level in dB (None = not playing, or a function
    of the seconds played so far in this call) and onsets per second per class. Returns every hop's Structure."""

    def __init__(self):
        self.layers, self.structure = LayerTracker("stems"), StructureTracker()
        self.phase = {name: 0.0 for name in CLASSES}
        self.heard = []                      # every Structure so far
        self.states = []                     # every Layers so far

    def play(self, seconds, drums=None, bass=None, vocals=None, other=None, kick=None, air=None, mix=None,
             hits=None, chord_step=0.0, lead_motion=0.0):
        out = []
        for i in range(int(round(seconds * 100))):
            t = i / 100
            at = lambda value: value(t) if callable(value) else value          # noqa: E731
            power = {"drums": P(at(drums)), "bass": P(at(bass)), "vocals": P(at(vocals)), "other": P(at(other))}
            mix_power = sum(power.values()) if mix is None else P(at(mix))
            events = {name: None for name in CLASSES}
            for name, rate in (hits or {}).items():
                self.phase[name] += at(rate) / 100
                if self.phase[name] >= 1:
                    self.phase[name] -= 1
                    events[name] = Onset(name, .8, 0.0)
            layers = self.layers.step(power, mix_power, P(at(kick)))
            self.states.append(layers)
            out.append(self.structure.step(layers, events, mix_power, P(at(air)), at(chord_step), at(lead_motion)))
        self.heard += out
        return out


def test_a_quiet_song_is_not_dim():
    for level in (-30, -10):
        heard = Band().play(10, drums=level, other=level)
        assert isinstance(heard[-1], Structure) and heard[-1].section in SECTIONS
        assert all(.55 <= s.energy <= .65 for s in heard[100:])


def test_a_crescendo_shows():
    band = Band()
    band.play(10, other=-24)
    band.play(10, other=lambda t: -24 + 1.2 * t)
    top = band.play(2, other=-12)
    assert max(s.energy for s in top) >= .9


def test_energy_settles_on_a_new_level():
    band = Band()
    band.play(10, other=-24)
    band.play(10, other=lambda t: -24 + 1.2 * t)
    later = band.play(62, other=-12)
    assert all(.5 <= s.energy <= .7 for s in later[-200:])


def test_silence_has_no_energy_and_starts_again():
    band = Band()
    band.play(10, drums=-10, other=-10)
    assert band.play(5)[-1].energy == 0
    again = band.play(6, drums=-30, other=-30)
    assert all(.55 <= s.energy <= .65 for s in again[400:])


# ---- tension ---------------------------------------------------------------------------------------

GROOVE = dict(drums=-12, kick=-15, other=-18, air=-40, mix=-11, hits={"kick": 2, "hats": 4})


def test_a_steady_groove_has_no_tension():
    heard = Band().play(20, **GROOVE)
    assert all(s.tension < .2 for s in heard[200:])


def test_a_roll_with_thinning_kick_builds():
    band = Band()
    band.play(20, **GROOVE)
    built = band.play(4, **{**GROOVE, "kick": lambda t: -15 - 3 * t, "hits": {"kick": 2, "hats": lambda t: 4 + 2 * t}})
    assert max(s.tension for s in built) > .6


def test_a_riser_and_a_swell_build():
    band = Band()
    band.play(20, **GROOVE)
    built = band.play(4, **{**GROOVE, "air": lambda t: -40 + 2.5 * t, "mix": lambda t: -11 + 1.5 * t})
    assert max(s.tension for s in built) > .6


def test_one_cue_alone_is_half_at_most():
    band = Band()
    band.play(20, **GROOVE)
    rolled = band.play(4, **{**GROOVE, "hits": {"kick": 2, "hats": lambda t: 4 + 2 * t}})
    rolled += band.play(4, **{**GROOVE, "hits": {"kick": 2, "hats": 12}})
    assert .3 < max(s.tension for s in rolled) <= .55


# ---- boundaries, drops, vocal entries ---------------------------------------------------------------

def marks(heard, field="boundary"):
    """Seconds into `heard` at which the event fired."""
    return [i / 100 for i, s in enumerate(heard) if getattr(s, field)]


def test_drums_entering_and_leaving_are_boundaries():
    band = Band()
    band.play(8, other=-12)
    playing = band.play(10, drums=-14, other=-12, hits={"hats": 4, "snare": 1})      # no low end: an entry, not a drop
    assert len(marks(playing)) == 1 and marks(playing)[0] <= .3 and playing[int(marks(playing)[0] * 100)].boundary == 1.0
    assert marks(playing, "drop") == []
    stopped = band.play(6, other=-12)
    assert len(marks(stopped)) == 1 and marks(stopped)[0] <= 1.5


def test_a_loudness_step_is_a_boundary():
    band = Band()
    band.play(8, other=-20)
    stepped = band.play(6, other=-11)
    assert len(marks(stepped)) == 1 and marks(stepped)[0] <= .6 and .5 <= max(s.boundary for s in stepped) <= 1
    wobbling = Band()
    wobbling.play(8, other=-20)
    for _ in range(10):
        assert marks(wobbling.play(1, other=-18) + wobbling.play(1, other=-22)) == []


def test_a_new_rhythm_is_a_boundary():
    band = Band()
    steady = band.play(14, drums=-12, kick=-15, other=-18, hits={"kick": 2, "hats": 2})
    assert marks(steady) == []                                   # music starting is not a boundary either
    changed = band.play(8, drums=-12, kick=-15, other=-18, hits={"kick": 2, "hats": 6})
    assert len(marks(changed)) == 1 and marks(changed)[0] <= 2.5 and max(s.boundary for s in changed) == .6


def test_bass_presence_alone_is_no_boundary():
    band = Band()
    band.play(8, drums=-12, kick=-15, other=-18, hits={"kick": 2, "hats": 4})
    for _ in range(5):
        band.play(2, drums=-12, kick=-15, other=-18, bass=-24, hits={"kick": 2, "hats": 4})       # 13 dB under the mix: there
        band.play(2, drums=-12, kick=-15, other=-18, bass=-31, hits={"kick": 2, "hats": 4})       # 20 dB under: gone
    flips = sum(a.bass.present != b.bass.present for a, b in zip(band.states, band.states[1:]))
    assert flips >= 8 and marks(band.heard) == [] and marks(band.heard, "drop") == []
    assert len({s.section for s in band.heard[800:]}) == 1


def test_a_drop_is_seen_once():
    band = Band()
    band.play(8, other=-20)
    hit = band.play(3, drums=-10, kick=-12, other=-20, mix=-12, hits={"kick": 2, "hats": 4})
    assert len(marks(hit, "drop")) == 1 and marks(hit, "drop")[0] <= 1.3
    at = int(marks(hit, "drop")[0] * 100)
    assert all(s.boundary == 0 for s in hit[at:at + 200])
    assert all(s.tension < .2 for s in hit[at:])                 # the drop releases the tension; its own jump is no new build
    again = band.play(3, drums=-2, kick=-2, other=-20, mix=-3, hits={"kick": 2, "hats": 4})
    assert marks(again, "drop") == []


def test_music_starting_is_not_a_drop():
    band = Band()
    band.play(3)
    started = band.play(8, drums=-10, kick=-12, bass=-14, other=-16, vocals=-12, hits={"kick": 2, "hats": 4})
    assert marks(started, "drop") == [] and marks(started) == []


def test_vocal_entries_and_exits():
    band = Band()
    band.play(8, other=-18)
    sung = band.play(5, other=-18, vocals=-15)
    assert len(marks(sung, "vocal_entry")) == 1 and marks(sung, "vocal_entry")[0] <= .3 and marks(sung, "vocal_exit") == []
    after = band.play(3, other=-18)
    assert len(marks(after, "vocal_exit")) == 1 and marks(after, "vocal_exit")[0] <= 1.5 and marks(after, "vocal_entry") == []


# ---- sections ----------------------------------------------------------------------------------------

def first(heard, section):
    """Seconds into `heard` at which the section is first this one (None if never)."""
    return next((i / 100 for i, s in enumerate(heard) if s.section == section), None)


def test_sections_follow_the_layers():
    band = Band()
    assert all(s.section == "quiet" for s in band.play(2))
    alone = band.play(8, other=-18, air=-40)
    assert first(alone, "breakdown") <= 1.2 and alone[-1].section == "breakdown"
    joined = band.play(12, drums=-14, other=-18, air=-40, mix=-12, hits={"hats": 4, "snare": 1})
    assert first(joined, "groove") <= 2.5 and joined[-1].section == "groove"
    rising = band.play(6, drums=-14, other=-18, air=lambda t: -40 + 2.5 * min(t, 4), mix=lambda t: -12 + 1.5 * min(t, 4),
                       hits={"hats": 4, "snare": 1})
    assert first(rising, "build") is not None and rising[-1].section == "build"
    dropped = band.play(10, drums=-4, kick=-6, other=-18, air=-40, mix=0, hits={"kick": 2, "hats": 4})
    at = marks(dropped, "drop")[0]
    assert first(dropped, "drop") == at and all(s.section == "drop" for s in dropped[int(at * 100):int(at * 100) + 400])
    assert 4 <= first(dropped[int(at * 100):], "groove") <= 8.5 and dropped[-1].section == "groove"


def test_a_short_dip_changes_nothing():
    band = Band()
    script = [.1] * 600 + [.65] * 300 + [.3] * 50 + [.65] * 150 + [.3] * 200
    band.structure._tension = lambda layers: script.pop(0)          # the section rule alone, on a tension we choose
    heard = band.play(13, drums=-12, kick=-15, other=-18, hits={"kick": 2, "hats": 4})
    assert heard[550].section == "groove" and 6.9 <= first(heard, "build") <= 7.1
    assert all(s.section == "build" for s in heard[720:1150])       # through the half-second dip
    assert heard[-1].section == "groove" and heard[1195].section == "build"       # a second under 0.4 ends it
    gap = Band()
    gap.play(8, drums=-12, kick=-15, other=-18, hits={"kick": 2, "hats": 4})
    gap.play(.6, other=-18)
    assert all(s.section == "groove" for s in gap.heard[300:] + gap.play(3, drums=-12, kick=-15, other=-18, hits={"kick": 2, "hats": 4}))


def test_a_quieter_song_is_not_quiet():
    band = Band()
    band.play(10, drums=-12, kick=-15, other=-18, hits={"kick": 2, "hats": 4})
    assert band.play(5)[-1].section == "quiet"
    soft = band.play(6, drums=-32, kick=-35, other=-38, hits={"kick": 2, "hats": 4})
    assert first(soft, "groove") <= 2.5 and soft[-1].section == "groove"


def test_since_restarts():
    band = Band()
    band.play(2)
    heard = band.play(8, other=-18) + band.play(8, drums=-14, other=-18, mix=-12, hits={"hats": 4})
    for a, b in zip(heard, heard[1:]):
        if b.section != a.section:
            assert b.since_s < .02
        else:
            assert b.since_s == pytest.approx(a.since_s + .01)


# ---- focus -------------------------------------------------------------------------------------------

BAND = dict(drums=-12, kick=-15, other=-13, vocals=-16, hits={"kick": 2, "hats": 2})      # all three groups present
DRUMS, HARMONY, LEAD = 0, 1, 2


def figure(structure):
    """Which group has the largest weight."""
    return max(range(3), key=lambda i: structure.focus[i])


def test_focus_sums_to_one():
    band = Band()
    band.play(3)
    band.play(8, other=-18)
    band.play(8, **BAND)
    band.play(6, **{**BAND, "hits": {"kick": 2, "hats": 6}}, lead_motion=.2, chord_step=40)
    band.play(5)
    assert all(abs(sum(s.focus) - 1) < 1e-9 and all(0 <= w <= 1 for w in s.focus) for s in band.heard)
    assert band.heard[100].focus == (1 / 3, 1 / 3, 1 / 3)             # nothing playing: no figure


def test_the_only_thing_changing_gets_the_focus():
    band = Band()
    band.play(20, **BAND)
    changed = band.play(4, **{**BAND, "hits": {"kick": 2, "hats": 6}})
    assert figure(changed[-1]) == DRUMS and changed[-1].focus[DRUMS] > .5


def test_a_moving_lead_gets_the_focus():
    band = Band()
    band.play(20, drums=-12, kick=-15, other=-18, hits={"kick": 2})
    moving = band.play(5, drums=-12, kick=-15, other=-18, hits={"kick": 2}, lead_motion=.3)
    assert all(figure(s) == LEAD and s.focus[LEAD] > .5 for s in moving[300:])       # it takes the focus from a kick that carries the mix


def test_a_chord_change_gets_the_focus():
    band = Band()
    still = band.play(20, other=-18, vocals=-16)
    assert still[-1].focus[DRUMS] < .05 and abs(still[-1].focus[HARMONY] - still[-1].focus[LEAD]) < .05
    changed = band.play(.5, other=-18, vocals=-16, chord_step=40) + band.play(1, other=-18, vocals=-16)
    assert figure(changed[-1]) == HARMONY and changed[-1].focus[HARMONY] > .5


def test_focus_fades():
    light = {**BAND, "kick": -30}                               # a kick too light to carry the mix: nobody stands out by default
    band = Band()
    band.play(20, **light)
    band.play(4, **{**light, "hits": {"kick": 2, "hats": 6}}, lead_motion=.2)
    later = band.play(16, **{**light, "hits": {"kick": 2, "hats": 6}})
    assert all(layer.present for layer in (band.states[-1].drums, band.states[-1].other, band.states[-1].vocals))
    assert all(w <= .5 for w in later[-1].focus) and all(abs(w - 1 / 3) < .05 for w in later[-1].focus)


# ---- found on real music -------------------------------------------------------------------

def test_a_drop_after_a_breakdown_stays_a_drop():
    band = Band()
    band.play(8, other=-20)                                  # a breakdown, wanted for a long time
    hit = band.play(6, drums=-10, kick=-12, other=-20, mix=-12, hits={"kick": 2, "hats": 4})
    at = int(marks(hit, "drop")[0] * 100)
    assert all(s.section == "drop" for s in hit[at:at + 400])


def test_a_near_silent_intro_does_not_pin_the_energy():
    band = Band()
    intro = band.play(6, other=-52)                          # audible, but under what counts as music
    assert intro[-1].energy == 0 and intro[-1].section == "quiet"
    loud = band.play(20, drums=-15, other=-18)
    assert all(.5 <= s.energy <= .75 for s in loud[1000:])


def test_big_hits_one_after_another_are_one_drop():
    band = Band()
    band.play(8, other=-20)
    hits = []
    for _ in range(4):
        hits += band.play(.3, drums=-10, kick=-12, other=-20, mix=-12) + band.play(2.7, other=-20)
    assert len(marks(hits, "drop")) == 1 and marks(hits, "drop")[0] <= .3


def test_drums_coming_and_going_are_one_boundary():
    band = Band()
    band.play(8, other=-12)
    flapping = []
    for _ in range(5):
        flapping += band.play(.5, drums=-14, other=-12, hits={"hats": 4}) + band.play(2.5, other=-12)
    assert sum(a.drums.present != b.drums.present for a, b in zip(band.states, band.states[1:])) == 10
    assert len(marks(flapping)) == 1 and marks(flapping)[0] <= .3      # the first entry; the rest is the same thing going on
    assert all(s.section == "breakdown" for s in flapping)            # and a few hits are not a groove


def test_phrases_are_not_boundaries():
    band = Band()
    band.play(8, other=-15)
    phrases = []
    for _ in range(6):                                       # a voice alone: two seconds sung, one second of breath
        phrases += band.play(1, other=-40) + band.play(2, other=-15)
    assert len(marks(phrases)) <= 1                          # the first breath is a change; the rest is the same thing going on
    louder = band.play(4, other=-8)                          # louder than anything in the last seconds: that is a step
    assert len(marks(louder)) == 1 and marks(louder)[0] <= .6


def test_a_kick_that_carries_the_mix_is_the_figure():
    band = Band()
    pounding = band.play(20, drums=-8, kick=-9, other=-12, hits={"kick": 2, "hats": 4})       # a steady pad under a loud kick
    assert band.states[-1].other.present and figure(pounding[-1]) == DRUMS and pounding[-1].focus[DRUMS] > .6
    sliding = band.play(6, drums=-8, kick=-9, other=-12, hits={"kick": 2, "hats": 4}, lead_motion=.3)
    assert figure(sliding[-1]) == LEAD                           # until a lead starts to travel
    soft = Band().play(20, drums=-14, kick=-26, other=-12, hits={"kick": 2, "hats": 4})        # a light kick under the same pad
    assert soft[-1].focus[DRUMS] < .55


def test_a_kick_that_came_in_during_the_first_seconds_is_no_drop_when_they_end():
    """Found on a ballad clip: one drum hit at 3.1 s was reported as a drop at 4.0 s, the hop the song's first
    4 s ended, because the jump was still standing. A drop is the jump happening, not the jump being there."""
    band = Band()
    start = band.play(3.1, other=-18)
    entered = band.play(6, drums=-10, kick=-12, other=-18, mix=-10, hits={"kick": 2, "hats": 4})
    assert marks(start + entered, "drop") == []
