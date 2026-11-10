"""The one record every look and the studio read: what the music is doing at time t."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from .layers import Layers
from .lead import Voice
from .levels import LEVELS
from .onsets import Onset
from .rhythm import BeatState
from .structure import Structure
from .tone import Tone

_ZERO_LEVELS = MappingProxyType({name: 0.0 for name in LEVELS})


@dataclass(frozen=True, slots=True)
class Features:
    t: float                              # host time of the hop's last sample
    kick: Onset | None = None
    snare: Onset | None = None
    hats: Onset | None = None
    bass: Onset | None = None
    vocal: Onset | None = None
    stab: Onset | None = None             # a chord or synth hit in the "other" layer; only ever read from stems
    levels: Mapping[str, float] = field(default_factory=lambda: _ZERO_LEVELS)
    beat: BeatState = field(default_factory=BeatState)
    silent: bool = False
    layers: Layers | None = None          # drums, bass, vocals, other: level and presence
    source: str = "bands"                 # where layers and events came from: "stems" or "bands"
    structure: Structure | None = None    # energy, tension, section, boundaries and drops, focus
    tone: Tone | None = None              # the harmony's colour on the circle of fifths
    voice: Voice | None = None            # the lead: where it sits in its range and how it moves

    @staticmethod
    def silence(t: float) -> "Features":
        return Features(t=t, beat=BeatState(t=t), silent=True)


__all__ = ["Features", "LEVELS"]
