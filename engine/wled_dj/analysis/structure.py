"""Where the song is, read from its layers every 10 ms: how much energy it has against its own usual level,
how much tension is building, which section it is in, where the boundaries and drops fall, when a voice
comes and goes, and which layer is the figure.

All thresholds are global. All times are counted in hops, never in beats."""
from __future__ import annotations

from collections import deque
from itertools import islice
from dataclasses import dataclass
import math
from typing import Mapping

from .frontend import HOP_S
from .layers import FLOOR_DB, Layers, Mean, power_db
from .onsets import Onset

SECTIONS = ("quiet", "groove", "breakdown", "build", "drop")
SILENT_DB = -60.0
RESTART_S = 3.0                      # this long silent, and what comes next is a new song
SETTLE_S = 4.0                       # a new song's first seconds: too little is known to call anything a change
LOUD_S, USUAL_S = 3.0, 30.0          # loudness now, and the usual loudness it is read against
USUAL_WARM_S = 5.0                   # for a song's first seconds the usual loudness is simply their average
ENERGY_USUAL, ENERGY_RANGE_DB = .6, 30.0
DRUM_HITS = ("kick", "snare", "hats")
# Tension: four cues, each 0-1. A rise is measured over RISE_S; a rate against the BEFORE_S before the last second.
RISE_S, BEFORE_S = 4.0, 8.0
ROLL_FLOOR = 2.0                     # onsets a second: a roll is judged against at least this much
ROLL_AFTER_S = 3.0                   # and only once the drums have played this long, against their own stay
THIN_FROM_DB, THIN_FULL_DB = 3.0, 12.0        # kick weight this far under its 8 s mean: cue 0, cue 1
RISER_FROM_DB, RISER_FULL_DB = 2.0, 10.0      # the air level's rise over 4 s
SWELL_FROM_DB, SWELL_FULL_DB = 1.0, 6.0       # the mix level's rise over 4 s
TENSION_UP_S, TENSION_DOWN_S = .5, 2.0
# Boundaries. Bass presence is never read: a third of bass-free time reads as bass.
BOUNDARY_GAP_S = 2.0                 # at most one boundary in this long; the first wins
STAY_S = 4.0                         # a layer coming or going is a boundary only if it had been away, or there, this long
PRESENCE_STRENGTH = {"drums": 1.0, "vocals": .7, "other": .5}
# A loudness step: the last 0.4 s are this much louder than anything, or quieter than everything, in the 4 s before
# them. (Not against an average: a voice alone steps up and down with every phrase, and that is no boundary.)
STEP_DB, STEP_AUDIBLE_DB = 5.0, -40.0
STEP_FAST_S, STEP_CLEAR_S, STEP_BACK_S = .4, .5, 4.5
RHYTHM_NOW_S, RHYTHM_BEFORE_S = 2.0, 6.0      # a class's onset rate now against the 6 s before
RHYTHM_FACTOR, RHYTHM_RATE, RHYTHM_HOLD_S, RHYTHM_STRENGTH = 2.0, 1.5, .5, .6
# A drop: kick weight far above the highest it was 2 to 10 s ago, with the mix getting louder. (Against the
# highest, not the average: big hits a few seconds apart are one event, not a drop each.)
DROP_JUMP_DB, DROP_KICK_DB, DROP_SWELL_DB, DROP_GAP_S = 10.0, -30.0, 2.0, 8.0
DROP_FROM_S, DROP_TO_S = 2.0, 10.0
# Sections. A change needs its reason to last SECTION_HOLD_S, except into a drop.
QUIET_DB = -45.0                     # the mix's 3 s level under this is quiet
HARMONY_REL_DB = -15.0               # bass counts as playing by its level against the mix, never by its presence flag
BUILD_ON, BUILD_OFF = .6, .4         # tension that starts a build, and that keeps one going
SECTION_HOLD_S = 1.0
GROOVE_HOLD_S = 2.0                  # drums must stay this long before a breakdown or a quiet ends: a few hits are no groove
DROP_MIN_S, DROP_MAX_S = 4.0, 8.0
# Focus: which group is the figure. A group that is merely there weighs FOCUS_THERE; one with news weighs up to 1 more.
FOCUS_FADE_S, FOCUS_SMOOTH_S, FOCUS_THERE = 4.0, 1.5, .15
CHORD_STEP_DEG = 15.0                # the chord colour moved this far within a second: the harmony changed
LEAD_MOVES, LEAD_SWEEPS = .15, .3    # of its range, travelled in 4 s: a lead that went this far is moving; this far, it is all news
# A kick that carries the mix keeps the drums in play without any news: its weight (the drums' low band over a
# second) against the mix, from 12 dB under it (nothing) to 3 dB under it (KICK_FIGURE).
KICK_FROM_DB, KICK_FULL_DB, KICK_FIGURE = -12.0, -3.0, .6
THIRDS = (1 / 3, 1 / 3, 1 / 3)


@dataclass(frozen=True, slots=True)
class Structure:
    energy: float = 0.0              # 0-1: loudness against the song's own running average; 0.6 is "as usual"
    tension: float = 0.0             # 0-1
    section: str = "quiet"
    since_s: float = 0.0             # how long the section has lasted
    boundary: float = 0.0            # 0, or the strength (0-1] of a boundary seen on this hop
    drop: bool = False               # true on the hop a drop is seen
    vocal_entry: bool = False
    vocal_exit: bool = False
    focus: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)     # drums, harmony (bass and other), lead; sums to 1


def _follow(seconds: float) -> float:
    """The share of the way a value moves toward its target in one hop, for a time constant of `seconds`."""
    return 1 - math.exp(-HOP_S / seconds)


def _hops(seconds: float) -> int:
    return round(seconds / HOP_S)


def _ramp(value: float, start: float, full: float) -> float:
    """0 at `start`, 1 at `full`, straight between."""
    return min(1.0, max(0.0, (value - start) / (full - start)))


class _Past:
    """A value's recent history: what it was `seconds` ago."""

    def __init__(self, seconds: float):
        self.values: deque[float] = deque(maxlen=_hops(seconds) + 1)

    def add(self, value: float) -> None:
        self.values.append(value)

    def ago(self, seconds: float) -> float | None:
        back = _hops(seconds)
        return self.values[-1 - back] if len(self.values) > back else None


class StructureTracker:
    def __init__(self):
        self.silent_hops = 0
        self._start()

    def _start(self) -> None:
        """Forget the song so far."""
        self.age = 0                                           # hops since the song's first sound
        self.loud = Mean(LOUD_S, so_far=True)
        self.started = False                                   # the song's first sound has been heard
        self.usual_db: float | None = None
        self.usual_hops = 0
        self.mix_1s, self.air_1s = Mean(1.0, so_far=True), Mean(1.0, so_far=True)
        self.mix_1s_db, self.air_1s_db = _Past(RISE_S), _Past(RISE_S)
        self.kick_usual = Mean(BEFORE_S, so_far=True)          # of kick weight, in dB
        self.hits: dict[str, deque[int]] = {name: deque() for name in DRUM_HITS}
        self.tension = 0.0
        self.present: dict[str, bool] | None = None            # drums, vocals, other as last seen
        self.stayed = {name: 0 for name in PRESENCE_STRENGTH}  # hops each has been in that state
        self.mix_fast = Mean(STEP_FAST_S, so_far=True)
        self.mix_fast_db = _Past(STEP_BACK_S)
        self.stepping = self.jumping = False
        self.rhythm_hops = {name: 0 for name in DRUM_HITS}     # how long each class's rate has stood changed
        self.kick_past = _Past(DROP_TO_S)
        self.last_boundary = self.last_drop = -math.inf        # in hops of this song
        self.loud_db = FLOOR_DB
        self.section, self.section_hops = "quiet", 0
        self.wanted, self.wanted_hops = "quiet", 0
        self.news = [0.0, 0.0, 0.0]                             # drums, harmony, lead: 1 when something happens, then fading
        self.happened = [False, False]                         # to the drums, to the harmony, on this hop
        self.focus = list(THIRDS)

    def step(self, layers: Layers, events: Mapping[str, Onset | None], mix_power: float, air_power: float,
             chord_step: float = 0.0, lead_motion: float = 0.0) -> Structure:
        """One hop. mix_power, air_power: mean power over the hop (a full-scale sine is 0.5), air being the
        other layer's 4-12 kHz. chord_step: degrees the chord colour moved in the last second.
        lead_motion: how far an audible lead travelled in its range over the last 4 s (LeadTracker.travel)."""
        self.silent_hops = self.silent_hops + 1 if power_db(mix_power) < SILENT_DB else 0
        if self.silent_hops and self.silent_hops % _hops(RESTART_S) == 0:
            self._start()
        if not self.started and self.silent_hops:
            return Structure()                                 # a song starts with its first sound
        self.started = True
        self.age += 1
        for name in DRUM_HITS:
            if events.get(name) is not None:
                self.hits[name].append(self.age)
            while self.hits[name] and self.hits[name][0] <= self.age - _hops(BEFORE_S + 2):
                self.hits[name].popleft()
        self.mix_1s_db.add(power_db(self.mix_1s.add(mix_power)))
        self.air_1s_db.add(power_db(self.air_1s.add(air_power)))
        boundary, drop, entry, exit_ = self._marks(layers, mix_power)
        if drop:
            self.tension = 0.0
        energy, tension = self._energy(mix_power), self._tension(layers)
        section = self._section(layers, tension, drop)
        focus = self._focus(layers, entry, chord_step, lead_motion)
        return Structure(energy, tension, section, self.section_hops * HOP_S, boundary, drop, entry, exit_, focus)

    def _energy(self, mix_power: float) -> float:
        loud_db = self.loud_db = power_db(self.loud.add(mix_power))
        if loud_db >= QUIET_DB:                      # the usual level is the music's, not that of a hush before or within it
            self.usual_hops += 1
            if self.usual_db is None:
                self.usual_db = loud_db
            warming = self.usual_hops <= _hops(USUAL_WARM_S)
            self.usual_db += (loud_db - self.usual_db) * (max(_follow(USUAL_S), 1 / self.usual_hops) if warming else _follow(USUAL_S))
        if self.usual_db is None or loud_db < SILENT_DB:
            return 0.0
        return min(1.0, max(0.0, ENERGY_USUAL + (loud_db - self.usual_db) / ENERGY_RANGE_DB))

    def _rate(self, names: tuple[str, ...], newest_s: float, oldest_s: float) -> float:
        """Onsets a second of these classes between `oldest_s` and `newest_s` ago (no further back than the song)."""
        newest, oldest = self.age - _hops(newest_s), max(0, self.age - _hops(oldest_s))
        if newest <= oldest:
            return 0.0
        return sum(1 for name in names for hop in self.hits[name] if oldest < hop <= newest) / ((newest - oldest) * HOP_S)

    def _tension(self, layers: Layers) -> float:
        kick_usual = self.kick_usual.add(layers.kick_weight_db - FLOOR_DB) + FLOOR_DB      # Mean holds values from 0 up
        cues = [0.0, 0.0, 0.0, 0.0]
        if self.age > _hops(SETTLE_S) and self.age - self.last_drop >= _hops(RISE_S):      # a drop's own jump is no build
            if layers.drums.present and layers.drums.since_s >= ROLL_AFTER_S:     # drums coming in are no roll
                before = self._rate(("snare", "hats"), 1.0, min(1.0 + BEFORE_S, layers.drums.since_s))
                cues[0] = _ramp(self._rate(("snare", "hats"), 0.0, 1.0) - before, 0.0, max(before, ROLL_FLOOR))
            if layers.drums.present:
                cues[1] = _ramp(kick_usual - layers.kick_weight_db, THIN_FROM_DB, THIN_FULL_DB)
            cues[2] = _ramp(self.air_1s_db.values[-1] - self.air_1s_db.ago(RISE_S), RISER_FROM_DB, RISER_FULL_DB)
            cues[3] = _ramp(self.mix_1s_db.values[-1] - self.mix_1s_db.ago(RISE_S), SWELL_FROM_DB, SWELL_FULL_DB)
        target = sum(sorted(cues)[-2:]) / 2
        self.tension += (target - self.tension) * _follow(TENSION_UP_S if target > self.tension else TENSION_DOWN_S)
        return self.tension

    def _marks(self, layers: Layers, mix_power: float) -> tuple[float, bool, bool, bool]:
        """(boundary strength, drop, vocal entry, vocal exit) for this hop."""
        settled = self.age > _hops(SETTLE_S)
        now = {name: layers[name].present for name in PRESENCE_STRENGTH}
        before, self.present = (self.present or now), now
        flipped = [name for name in now if now[name] != before[name]]
        lasting = [name for name in flipped if self.stayed[name] >= _hops(STAY_S)]      # the rest is coming and going
        for name in now:
            self.stayed[name] = 0 if name in flipped else self.stayed[name] + 1
        candidates = [PRESENCE_STRENGTH[name] for name in lasting]

        fast_db = power_db(self.mix_fast.add(mix_power))
        older = len(self.mix_fast_db.values) - _hops(STEP_CLEAR_S)         # how many values are clear of the last 0.4 s
        step = 0.0
        if older > 0:
            highest, lowest = max(islice(self.mix_fast_db.values, 0, older)), min(islice(self.mix_fast_db.values, 0, older))
            if max(fast_db, highest) > STEP_AUDIBLE_DB:
                step = max(fast_db - highest, lowest - fast_db, 0.0)
        self.mix_fast_db.add(fast_db)
        stepped = step >= STEP_DB and not self.stepping
        if stepped:
            candidates.append(min(1.0, max(.5, step / 10)))
        self.stepping = step >= STEP_DB

        whole = RHYTHM_NOW_S + RHYTHM_BEFORE_S                  # the comparison must lie inside the drums' stay
        new_rhythm = False
        for name in DRUM_HITS:
            rates = (self._rate((name,), 0.0, RHYTHM_NOW_S), self._rate((name,), RHYTHM_NOW_S, whole))
            changed = (layers.drums.present and layers.drums.since_s >= whole and self.age - self.last_drop >= _hops(whole)
                       and max(rates) >= RHYTHM_RATE and max(rates) >= RHYTHM_FACTOR * min(rates))
            self.rhythm_hops[name] = self.rhythm_hops[name] + 1 if changed else 0
            if self.rhythm_hops[name] == _hops(RHYTHM_HOLD_S):
                candidates.append(RHYTHM_STRENGTH)
                new_rhythm = True

        older = len(self.kick_past.values) - _hops(DROP_FROM_S)            # how many values are at least 2 s old
        kick_before = max(islice(self.kick_past.values, 0, older)) if older > 0 else None
        self.kick_past.add(layers.kick_weight_db)
        mix_then = self.mix_1s_db.ago(DROP_FROM_S)
        jumped = (kick_before is not None and mix_then is not None
                  and layers.kick_weight_db > DROP_KICK_DB
                  and layers.kick_weight_db - kick_before >= DROP_JUMP_DB
                  and self.mix_1s_db.values[-1] - mix_then >= DROP_SWELL_DB)
        # A drop is the jump happening: one that began in the song's first seconds is not a drop when they end.
        drop = jumped and not self.jumping and settled and self.age - self.last_drop >= _hops(DROP_GAP_S)
        self.jumping = jumped
        self.happened = [("drums" in lasting and now["drums"]) or new_rhythm or drop, stepped and "drums" not in flipped]
        boundary = 0.0
        if drop:
            self.last_drop = self.last_boundary = self.age
        elif candidates and settled and self.age - self.last_boundary >= _hops(BOUNDARY_GAP_S):
            boundary, self.last_boundary = max(candidates), self.age
        return boundary, drop, settled and "vocals" in flipped and now["vocals"], settled and "vocals" in flipped and not now["vocals"]

    def _section(self, layers: Layers, tension: float, drop: bool) -> str:
        self.section_hops += 1
        if drop:
            self.section, self.section_hops = "drop", 0
            self.wanted, self.wanted_hops = "drop", 0          # whatever was wanted before the drop starts its hold again
            return self.section
        playing = (layers.drums.present or layers.vocals.present or layers.other.present
                   or layers.bass.rel_db > HARMONY_REL_DB)
        if self.loud_db < QUIET_DB or not playing:
            wanted = "quiet"
        elif not layers.drums.present:
            wanted = "breakdown"
        elif tension >= BUILD_ON or (self.section == "build" and tension >= BUILD_OFF):
            wanted = "build"
        else:
            wanted = "groove"
        self.wanted_hops = self.wanted_hops + 1 if wanted == self.wanted else 1
        self.wanted = wanted
        if wanted != self.section:
            arriving = wanted in ("groove", "build") and self.section in ("quiet", "breakdown")
            held = self.wanted_hops >= _hops(GROOVE_HOLD_S if arriving else SECTION_HOLD_S)
            if self.section == "drop":             # a drop lasts 4 to 8 s; only the music thinning out cuts it short
                held = (held and (wanted in ("quiet", "breakdown") or self.section_hops >= _hops(DROP_MIN_S))
                        or self.section_hops >= _hops(DROP_MAX_S))
            if held:
                self.section, self.section_hops = wanted, 0
        return self.section

    def _focus(self, layers: Layers, vocal_entry: bool, chord_step: float, lead_motion: float) -> tuple[float, float, float]:
        fade = math.exp(-HOP_S / FOCUS_FADE_S)
        carrying = KICK_FIGURE * _ramp(layers.kick_weight_db - self.mix_1s_db.values[-1], KICK_FROM_DB, KICK_FULL_DB)
        drums = 1.0 if self.happened[0] else max(self.news[0] * fade, carrying)
        harmony = 1.0 if self.happened[1] or chord_step >= CHORD_STEP_DEG else self.news[1] * fade
        lead = max(self.news[2] * fade, 1.0 if vocal_entry else 0.0,
                   min(1.0, lead_motion / LEAD_SWEEPS) if lead_motion >= LEAD_MOVES else 0.0)
        self.news = [drums, harmony, lead]
        there = (layers.drums.present, layers.other.present or layers.bass.rel_db > HARMONY_REL_DB,
                 layers.vocals.present or lead_motion >= LEAD_MOVES)
        weights = [FOCUS_THERE * here + news for here, news in zip(there, self.news)]
        total = sum(weights)
        target = [w / total for w in weights] if total > 0 else THIRDS
        ease = _follow(FOCUS_SMOOTH_S)
        self.focus = [f + (t - f) * ease for f, t in zip(self.focus, target)]
        return (self.focus[0], self.focus[1], self.focus[2])
