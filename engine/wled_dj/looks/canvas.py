"""What a look paints on: one row of pixels held as lightness and colour (OKLab), converted once at the end.

Positions: `u` runs 0-1 from one end of the strip to the other; `r` is the distance from the centre, 0 there
and 1 at both ends, so anything drawn as a function of `r` is mirror-symmetric and a shape `w` wide in `r`
covers the share `w` of the strip. Shapes are weights 0-1 per pixel and never narrower than one pixel."""
from __future__ import annotations

import math

import numpy as np

from .colour import _polar, encode, fit


def pulse(age_s: float, hold_s: float, decay_s: float) -> float:
    """An accent's envelope: full at once, held, then dying away. 0 before it starts."""
    if age_s < 0:
        return 0.0
    return 1.0 if age_s < hold_s else math.exp(-(age_s - hold_s) / decay_s)


def _pixel(r: np.ndarray) -> float:
    return 2.0 / len(r)                      # one pixel's length in r


def mass(r: np.ndarray, width: float, edge: float) -> np.ndarray:
    """A soft-edged mass at the centre, `width` wide (full inside, falling to nothing over `edge` beyond)."""
    width = max(width, _pixel(r))
    if edge <= 0:
        return (r < width).astype(np.float64)
    return np.clip((width + edge - r) / edge, 0.0, 1.0)


def band(r: np.ndarray, centre: float, width: float, edge: float) -> np.ndarray:
    """A band `width` wide around the distance `centre` from the middle of the strip, on both sides."""
    half = max(width, _pixel(r)) / 2
    centre = min(max(centre, half), 1.0 - half)          # a band at the very middle or the very end stays whole
    away = np.abs(r - centre)
    if edge <= 0:
        return (away <= half + 1e-12).astype(np.float64)
    return np.clip((half + edge - away) / edge, 0.0, 1.0)


class Canvas:
    def __init__(self, pixels: int):
        self.u = (np.arange(pixels) + .5) / pixels
        self.r = np.abs(2 * np.arange(pixels) + 1 - pixels) / pixels      # from whole numbers, so both halves are exactly alike
        self.left = self.u < .5
        self.lab = np.zeros((pixels, 3))

    def clear(self) -> None:
        self.lab[:] = 0.0

    def fill(self, lightness, chroma, hue) -> None:
        """Every pixel to this colour (scalars, or one value per pixel)."""
        self.lab[:] = _polar(lightness, chroma, hue)

    def paint(self, weight, lightness, chroma, hue) -> None:
        """Move each pixel toward this colour by its weight (0 leaves it, 1 replaces it)."""
        weight = np.clip(np.asarray(weight, np.float64), 0.0, 1.0)[:, None]
        self.lab += (_polar(lightness, chroma, hue) - self.lab) * weight

    def dim(self, factor) -> None:
        self.lab *= np.asarray(factor, np.float64)[..., None] if np.ndim(factor) else factor

    @property
    def lightness(self) -> np.ndarray:
        return self.lab[:, 0]

    def values(self) -> np.ndarray:
        """(pixels, 3) float32: what a look returns."""
        chroma = np.hypot(self.lab[:, 1], self.lab[:, 2])
        hue = np.degrees(np.arctan2(self.lab[:, 2], self.lab[:, 1]))
        return encode(fit(self.lab[:, 0], chroma, hue)).astype(np.float32)
