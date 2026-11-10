"""Colour as the eye sees it, for the looks.

A look thinks in lightness, chroma and hue (OKLab and its polar form OKLCH, Björn Ottosson's): equal steps
look equal, a fade through it stays clean, and two colours of one lightness look equally bright. That is
converted once: to linear light (sRGB primaries), and from there to the 0-1 values a look returns, which are
linear light to the power 1 / 2.2 (the compositor's gamma undoes it for the LEDs; a screen shows them as they are).

The ring is the palette every look takes its colours from, by position: a closed loop of eight anchors,
violet, magenta, rose, red-orange, orange, yellow, azure and blue, so that opposite positions contrast. Between
yellow and azure the loop crosses green. In good light that is a green; in dim light greens and cyans look grey
and muddy, so there the loop holds the yellow, lets its colour fall to nothing, and comes up again as azure."""
from __future__ import annotations

import numpy as np

from ..flash import luminance  # noqa: F401  (relative luminance belongs with the definition of a flash)

GAMMA = 2.2
_TO_LMS = np.array([[.4122214708, .5363325363, .0514459929],
                    [.2119034982, .6806995451, .1073969566],
                    [.0883024619, .2817188376, .6299787005]])
_TO_LAB = np.array([[.2104542553, .7936177850, -.0040720468],
                    [1.9779984951, -2.4285922050, .4505937099],
                    [.0259040371, .7827717662, -.8086757660]])
_FROM_LAB = np.array([[1.0, .3963377774, .2158037573],
                      [1.0, -.1055613458, -.0638541728],
                      [1.0, -.0894841775, -1.2914855480]])
_FROM_LMS = np.array([[4.0767416621, -3.3077115913, .2309699292],
                      [-1.2684380046, 2.6097574011, -.3413193965],
                      [-.0041960863, -.7034186147, 1.7076147010]])


def linear_to_oklab(rgb) -> np.ndarray:
    """(..., 3) linear light -> (..., 3) lightness, a, b."""
    return np.cbrt(np.asarray(rgb, np.float64) @ _TO_LMS.T) @ _TO_LAB.T


def oklab_to_linear(lab) -> np.ndarray:
    """(..., 3) lightness, a, b -> (..., 3) linear light; outside 0-1 where the colour is outside the gamut."""
    return ((np.asarray(lab, np.float64) @ _FROM_LAB.T) ** 3) @ _FROM_LMS.T


def encode(linear) -> np.ndarray:
    """Linear light -> the values a look returns."""
    return np.clip(linear, 0.0, 1.0) ** (1 / GAMMA)


def decode(values) -> np.ndarray:
    """A look's values -> linear light."""
    return np.clip(values, 0.0, 1.0) ** GAMMA


def _polar(lightness, chroma, hue_deg) -> np.ndarray:
    rad = np.radians(hue_deg)
    return np.stack(np.broadcast_arrays(lightness, chroma * np.cos(rad), chroma * np.sin(rad)), axis=-1)


def _gamut_table() -> np.ndarray:
    """The largest chroma inside the gamut for 101 lightnesses (rows) and 360 hues (columns), by bisection."""
    lightness = np.linspace(0, 1, 101)[:, None] * np.ones((1, 360))
    hue = np.arange(360.0)[None, :] * np.ones((101, 1))
    low, high = np.zeros((101, 360)), np.full((101, 360), .4)
    for _ in range(20):
        middle = (low + high) / 2
        rgb = oklab_to_linear(_polar(lightness, middle, hue))
        inside = (rgb.min(axis=-1) >= 0) & (rgb.max(axis=-1) <= 1)
        low, high = np.where(inside, middle, low), np.where(inside, high, middle)
    return low


_MAX_CHROMA = _gamut_table()


def max_chroma(lightness, hue_deg) -> np.ndarray:
    """The most chroma the LEDs can show at this lightness and hue."""
    at = np.clip(np.asarray(lightness, np.float64), 0.0, 1.0) * 100
    row = np.minimum(at.astype(int), 99)
    f = at - row
    hue = np.asarray(hue_deg, np.float64) % 360
    column = hue.astype(int) % 360
    g = hue - np.floor(hue)
    after = (column + 1) % 360
    table = _MAX_CHROMA
    return ((1 - f) * ((1 - g) * table[row, column] + g * table[row, after])
            + f * ((1 - g) * table[row + 1, column] + g * table[row + 1, after]))


def fit(lightness, chroma, hue_deg) -> np.ndarray:
    """(..., 3) linear light for this lightness, chroma and hue. A colour the LEDs cannot show keeps its
    lightness and its hue and loses chroma."""
    lightness, chroma, hue_deg = np.broadcast_arrays(np.asarray(lightness, np.float64), np.asarray(chroma, np.float64),
                                                     np.asarray(hue_deg, np.float64))
    lightness = np.clip(lightness, 0.0, 1.0)
    chroma = np.clip(chroma, 0.0, max_chroma(lightness, hue_deg))
    return np.clip(oklab_to_linear(_polar(lightness, chroma, hue_deg)), 0.0, 1.0)


DIM_ALL, DIM_NONE = .30, .45          # lightness at and under which no green is shown, and from which it is shown as it is
GREEN = (110.0, 215.0)                # the hues that look grey when dim
SEAM = .15                            # of a leg, either side of its middle: where the dim loop has no colour


class Ring:
    """A closed loop of anchors (hue in degrees, most chroma), at positions 0, 1/n, ... ; positions wrap at 1."""

    def __init__(self, anchors: list[tuple[float, float]]):
        self.hues = np.array([hue for hue, _ in anchors], np.float64)
        self.chromas = np.array([chroma for _, chroma in anchors], np.float64)
        self.steps = (np.roll(self.hues, -1) - self.hues + 180) % 360 - 180          # the short way to the next anchor
        along = (self.hues[:, None] + self.steps[:, None] * np.linspace(0, 1, 33)[None, :]) % 360
        self.green = ((along > GREEN[0]) & (along < GREEN[1])).any(axis=1)           # legs that cross green

    def colour(self, position, lightness, chroma: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(lightness, chroma, hue in degrees) at a place on the ring. `chroma` is a share of what the anchor
        allows and the LEDs can show."""
        position, lightness = np.broadcast_arrays(np.asarray(position, np.float64), np.asarray(lightness, np.float64))
        at = (position % 1.0) * len(self.hues)
        leg = at.astype(int) % len(self.hues)
        q = at - np.floor(at)
        after = (leg + 1) % len(self.hues)
        hue = np.radians(self.hues[leg] + self.steps[leg] * q)
        most = self.chromas[leg] + (self.chromas[after] - self.chromas[leg]) * q
        a, b = most * np.cos(hue), most * np.sin(hue)
        # The dim loop on a leg that crosses green: the first anchor, nothing, the second anchor.
        first = q < .5
        held = np.radians(np.where(first, self.hues[leg], self.hues[after]))
        left = np.where(first, self.chromas[leg], self.chromas[after]) * np.clip(np.abs(q - .5) / SEAM, 0.0, 1.0)
        dim_a = np.where(self.green[leg], left * np.cos(held), a)
        dim_b = np.where(self.green[leg], left * np.sin(held), b)
        good = np.clip((lightness - DIM_ALL) / (DIM_NONE - DIM_ALL), 0.0, 1.0)
        a, b = dim_a + (a - dim_a) * good, dim_b + (b - dim_b) * good
        degrees = np.degrees(np.arctan2(b, a)) % 360
        return lightness, chroma * np.minimum(np.hypot(a, b), max_chroma(lightness, degrees)), degrees


RING = Ring([(300, .26), (330, .26), (5, .24), (35, .22), (65, .18), (95, .16), (235, .18), (265, .26)])
