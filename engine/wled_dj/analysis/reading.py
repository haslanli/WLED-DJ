"""What one hop of separated stems says: its events, its layers, and from those the song's structure, the
harmony's colour and the lead voice. The detectors and trackers are the ones the engine runs on the mix,
each on its own stem."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..stems.link import STEM_HOP, StemHop
from ..stems.protocol import RATE as STEM_RATE
from .frontend import FULL_SCALE_ENERGY, FrontEnd, band
from .layers import LAYERS, Layers, LayerTracker
from .lead import LeadTracker, Voice
from .onsets import STEM_SPECS, Onset, OnsetDetector
from .structure import Structure, StructureTracker
from .tone import Tone, ToneTracker

AIR_HZ = (4000.0, 12000.0)           # where a riser lives


@dataclass(frozen=True, slots=True)
class Reading:
    events: dict[str, Onset | None]
    layers: Layers
    structure: Structure
    tone: Tone
    voice: Voice


def lead_motion(tracker: LeadTracker, voice: Voice) -> float:
    """How far an audible lead has travelled lately, for the structure's focus."""
    return tracker.travel if voice.level > 0 else 0.0


class LayerAnalyzer:
    """Separated stems in, a Reading out."""

    def __init__(self, specs=STEM_SPECS):
        self.fronts = {stem: FrontEnd(STEM_RATE, STEM_HOP, 1) for stem in LAYERS}
        self.detectors = [(stem, OnsetDetector(spec, STEM_RATE)) for stem, spec in specs]
        self.tracker = LayerTracker("stems")
        self.tone, self.lead, self.structure = ToneTracker(STEM_RATE), LeadTracker(STEM_RATE), StructureTracker()
        self.air = band(*AIR_HZ, STEM_RATE)

    def step(self, hop: StemHop) -> Reading:
        spectra = {stem: front.push(getattr(hop, stem)) for stem, front in self.fronts.items()}
        events = {detector.spec.name: detector.step(spectra[stem]) for stem, detector in self.detectors}
        power = {name: float(np.mean(np.square(getattr(hop, name), dtype=np.float64))) for name in LAYERS}
        mix = float(np.mean(np.square(hop.mix, dtype=np.float64)))
        layers = self.tracker.step(power, mix, spectra["drums"].low_power)
        tone = self.tone.step(hop.bass + hop.other + hop.vocals)
        voice = self.lead.step(spectra["vocals"], spectra["other"], layers)
        air = float(np.square(spectra["other"].mag[self.air], dtype=np.float64).sum()) * .5 / FULL_SCALE_ENERGY
        structure = self.structure.step(layers, events, mix, air, self.tone.chord_step, lead_motion(self.lead, voice))
        return Reading(events, layers, structure, tone, voice)
