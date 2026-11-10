"""Turns captured audio blocks into one Features record per 10 ms hop.

The mix is always analysed, as before. With a stem link, every hop is also sent for separation, and:
- once 100 stem hops have been read (warm-up), the layers come from the stems, and with them the song's
  structure, the harmony's colour and the lead voice;
- a hop whose stems are back in time takes its events from the stems too;
- a hop whose stems are late takes its events from the mix and keeps the last stems reading, so the lights
  never wait longer than they were told to. A block that holds two hops may wait 10 ms longer for the second.
Stabs (chords and synth hits in the "other" layer) are read from stems only: the mix cannot tell them from a
voice, so a hop that follows the mix has none.

Structure, colour and lead are also read from the mix all the time, for the hops that stems do not cover.
A boundary, drop or vocal entry seen in stems that came late is passed on once, with the next hop. A change
between the two sources is itself never an event: for two seconds after it no event is passed on, and the
section stays what it was.

In stems mode a hop's time is `latency_s` earlier than the mix hop that completed it: that is how far a
stem trails the audio it was made from."""
from __future__ import annotations

from dataclasses import replace
import math
from types import MappingProxyType

import numpy as np

from ..capture import AudioBlock
from ..stems.link import StemLink
from .features import Features
from .frontend import FULL_SCALE_ENERGY, HOP, HOP_S, RATE, FrontEnd, band
from .layers import LayerTracker, bands_power
from .lead import LeadTracker
from .levels import Levels
from .onsets import SPECS, STEM_SPECS, Onset, OnsetDetector
from .reading import AIR_HZ, LayerAnalyzer, Reading, lead_motion
from .rhythm import BeatTracker
from .structure import Structure, StructureTracker
from .tone import ToneTracker

GAP_RESET_S = .25
BEAT_SOURCES = ("kick", "snare", "hats")
WARM_HOPS = 100           # stem hops read before the stems are trusted (1 s)
QUIET_HOPS = 100          # silent hops after which nothing is sent for separation (1 s)
REFRACTORY_S = {spec.name: spec.refractory_s for spec in (*SPECS, *(spec for _, spec in STEM_SPECS))}
SOURCE_SETTLE_S = 2.0     # after the source of the layers changed: no structure events, and the section holds
AIR = band(*AIR_HZ)
NO_EVENTS = dict(boundary=0.0, drop=False, vocal_entry=False, vocal_exit=False)


class Analyzer:
    def __init__(self, stems: StemLink | None = None):
        self.stems = stems
        self.quiet_hops = 0
        self.idle = False          # true while silence keeps audio from being sent for separation
        self._reset()
        self.expected_next: float | None = None

    def _reset(self) -> None:
        self.front = FrontEnd()
        self.detectors = {spec.name: OnsetDetector(spec) for spec in SPECS}
        self.tracker = BeatTracker()
        self.levels = Levels()
        self.band_layers = LayerTracker("bands")
        self.band_tone, self.band_lead, self.band_structure = ToneTracker(RATE), LeadTracker(RATE), StructureTracker()
        self.pending = np.empty((0, 2), np.float32)
        self.source: str | None = None             # of the last Features that left
        self.settled_at = -math.inf                # until then a change of source keeps events back
        self.section: tuple[str, float] | None = None
        self._reset_stems()

    def _reset_stems(self) -> None:
        self.layer_analyzer = LayerAnalyzer()
        self.stem_hops_seen = 0
        self.stem_reading: Reading | None = None
        self.latched = dict(NO_EVENTS)             # structure events of stem hops not yet passed on
        self.last_event = {name: -math.inf for name in REFRACTORY_S}

    def feed(self, block: AudioBlock, wait_s: float = 0.0) -> list[Features]:
        """wait_s: how long this call may still wait for the stems of the block's first hop (negative when
        that time has passed). Every further hop in the block is 10 ms newer and may take 10 ms longer."""
        if self.expected_next is not None and abs(block.host_time - self.expected_next) > GAP_RESET_S:
            self._reset()
        self.expected_next = block.host_time + len(block.samples) / RATE
        start = block.host_time - len(self.pending) / RATE   # time of the first pending sample
        self.pending = np.concatenate((self.pending, block.samples.astype(np.float32, copy=False)))
        out = []
        used = 0
        while len(self.pending) - used >= HOP:
            hop = self.pending[used:used + HOP]
            t = start + (used + HOP - 1) / RATE
            features = self._process(hop, t)
            out.append((features, self._send(hop, t, features.silent)))
            used += HOP
        self.pending = self.pending[used:]
        if self.stems is None:
            return [self._settle(features) for features, _ in out]
        ready = self._collect(max(0.0, wait_s + (len(out) - 1) * HOP_S)) if out else {}
        return [self._settle(self._choose(features, index, ready)) for features, index in out]

    def _process(self, hop: np.ndarray, t: float) -> Features:
        spectrum = self.front.push(hop)
        onsets = {name: detector.step(spectrum) for name, detector in self.detectors.items()}
        strength = 0.0
        for name in BEAT_SOURCES:
            detector = self.detectors[name]
            reference = float(np.median(detector.history)) if detector.history else 0.0
            strength += detector.novelty / (reference + detector.floor)
        beat = self.tracker.step(t, strength, onsets["kick"])
        levels, silent = self.levels.step(spectrum, hop)
        power, mix_power, kick_power = bands_power(spectrum)
        layers = self.band_layers.step(power, mix_power, kick_power)
        tone = self.band_tone.step(hop.mean(axis=1))
        voice = self.band_lead.step(spectrum, spectrum, layers)
        air = float(np.square(spectrum.harm[AIR], dtype=np.float64).sum()) * .5 / FULL_SCALE_ENERGY
        structure = self.band_structure.step(layers, onsets, mix_power, air, self.band_tone.chord_step, lead_motion(self.band_lead, voice))
        return Features(t=t, levels=MappingProxyType(levels), beat=beat, silent=silent, layers=layers,
                        structure=structure, tone=tone, voice=voice, **onsets)

    # ---- stems -----------------------------------------------------------------------------
    def _send(self, hop: np.ndarray, t: float, silent: bool) -> int | None:
        """Send one hop for separation. Returns its index there, or None when it was not sent."""
        if self.stems is None:
            return None
        self.quiet_hops = self.quiet_hops + 1 if silent else 0
        if self.quiet_hops > QUIET_HOPS:
            self.idle = True                          # the helper idles through silence
            return None
        if self.idle:
            self.idle = False
            self._reset_stems()                       # sound is back: warm up again
        index = self.stems.push(hop, t)
        if index == 0:
            self._reset_stems()                       # the helper has just (re)started
        return index

    def _collect(self, wait_s: float) -> dict[int, tuple[float, Reading]]:
        ready = {}
        for hop in self.stems.collect(wait_s):
            reading = self.layer_analyzer.step(hop)
            self.stem_hops_seen += 1
            self.stem_reading = reading
            seen = reading.structure
            self.latched = dict(boundary=max(self.latched["boundary"], seen.boundary), drop=self.latched["drop"] or seen.drop,
                                vocal_entry=self.latched["vocal_entry"] or seen.vocal_entry,
                                vocal_exit=self.latched["vocal_exit"] or seen.vocal_exit)
            ready[hop.index] = (hop.t, reading)
        return ready

    def _choose(self, mix: Features, index: int | None, ready: dict) -> Features:
        events = {name: getattr(mix, name) for name in REFRACTORY_S}
        trusted = (index is not None and self.stems.state == "on" and self.stem_hops_seen >= WARM_HOPS
                   and self.stem_reading is not None)
        if not trusted:
            return replace(mix, **self._once(events, mix.t))
        reading = self.stem_reading                   # the latest: layers, structure, colour and lead hold while stems are late
        if index in ready:
            t, on_time = ready[index]
            events = {name: onset if onset is None or name not in self.detectors
                      else replace(onset, pan=self.detectors[name].pan) for name, onset in on_time.events.items()}
        else:                                         # the stems are late: events from the mix
            t = mix.t - self.stems.latency_s
        structure, self.latched = replace(reading.structure, **self.latched), dict(NO_EVENTS)
        return replace(mix, t=t, layers=reading.layers, structure=structure, tone=reading.tone, voice=reading.voice,
                       source="stems", **self._once(events, t))

    def _settle(self, features: Features) -> Features:
        """A change of source is not an event: for two seconds after one, structure events are kept back and
        the section stays what it was."""
        now = features.t if features.source == "bands" else features.t + self.stems.latency_s      # the mix's clock
        if self.source is not None and features.source != self.source:
            self.settled_at = now + SOURCE_SETTLE_S
        self.source = features.source
        if now < self.settled_at and self.section is not None:
            name, since = self.section
            self.section = (name, since + HOP_S)
            return replace(features, structure=replace(features.structure, section=name, since_s=since + HOP_S, **NO_EVENTS))
        self.section = (features.structure.section, features.structure.since_s)
        return features

    def _once(self, events: dict[str, Onset | None], t: float) -> dict[str, Onset | None]:
        """Drop an event that repeats its class within the class's refractory time, whichever source saw it."""
        for name, onset in events.items():
            if onset is None:
                continue
            if t - self.last_event[name] < REFRACTORY_S[name] - 1e-9:
                events[name] = None
            else:
                self.last_event[name] = t
        return events
