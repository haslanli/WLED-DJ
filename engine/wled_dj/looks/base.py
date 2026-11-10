"""A look turns Features into colours. Looks render in [0, 1]; the compositor handles output."""
from __future__ import annotations

from typing import ClassVar, Protocol

import numpy as np

from ..analysis.features import Features


class Look(Protocol):
    id: ClassVar[str]
    name: ClassVar[str]

    def __init__(self, pixels: int) -> None: ...

    def render(self, f: Features, lead: float) -> np.ndarray:
        """(pixels, 3) float32 in [0, 1]. f.t is the clock. lead = seconds the lights run late;
        beat-locked elements use f.beat.phase_at(f.t + lead). Reactive elements ignore it."""
        ...
