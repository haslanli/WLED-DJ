"""The canvas a look paints on, and the shapes it paints with."""
import math

import numpy as np
import pytest

from wled_dj.looks.canvas import Canvas, band, mass, pulse
from wled_dj.looks.colour import RING, decode, linear_to_oklab


def lab_of(canvas):
    return linear_to_oklab(decode(canvas.values().astype(np.float64)))


def test_a_pulse_is_full_at_once_and_holds():
    assert pulse(0.0, .05, .11) == 1 and pulse(.049, .05, .11) == 1
    assert abs(pulse(.05 + .11, .05, .11) - 1 / math.e) < 1e-6
    assert pulse(-.01, .05, .11) == 0
    assert pulse(5.0, .05, .11) < 1e-9


@pytest.mark.parametrize("pixels", [30, 79, 300])
def test_shapes_are_symmetric_and_never_vanish(pixels):
    canvas = Canvas(pixels)
    assert canvas.r.min() < 2 / pixels and abs(canvas.r.max() - (1 - 1 / pixels)) < 1e-9
    for shape in (mass(canvas.r, .1, .05), band(canvas.r, .5, .08, .04), mass(canvas.r, .001, 0.0),
                  band(canvas.r, .5, .001, 0.0), band(canvas.r, 1.0, .001, 0.0), band(canvas.r, 0.0, .001, 0.0)):
        assert shape.shape == (pixels,) and shape.min() >= 0 and shape.max() == 1
        assert np.allclose(shape, shape[::-1])
        assert (shape[canvas.left] == 1).any() or pixels % 2                # something is lit on each side
    wide = mass(canvas.r, .1, 0.0)
    assert abs(wide.mean() - .1) <= 1.5 / pixels                            # a mass 0.1 wide covers a tenth of the strip
    assert abs(band(canvas.r, .5, .08, 0.0).mean() - .08) <= 2.5 / pixels


def test_a_fade_is_even_in_lightness():
    for weight in (0.0, .25, .5, .75, 1.0):
        canvas = Canvas(78)
        canvas.paint(np.full(78, weight), 1.0, 0.0, 0.0)
        assert np.all(np.abs(lab_of(canvas)[:, 0] - weight) < .01)


def test_equal_lightness_looks_equally_bright():
    for k in range(8):
        canvas = Canvas(78)
        canvas.fill(*RING.colour(k / 8, .5))
        assert np.all(np.abs(lab_of(canvas)[:, 0] - .5) < .01), k


def test_paint_blends_toward_the_colour():
    canvas = Canvas(78)
    canvas.fill(*RING.colour(0.0, .3))
    before = canvas.lightness.copy()
    weight = mass(canvas.r, .1, 0.0)
    canvas.paint(weight, .9, .05, 300.0)
    assert np.allclose(canvas.lightness[weight == 1], .9) and np.allclose(canvas.lightness[weight == 0], before[weight == 0])
    canvas.dim(.5)
    assert np.allclose(canvas.lightness[weight == 1], .45)


def test_black_is_black():
    canvas = Canvas(78)
    assert not canvas.values().any()
    canvas.fill(.4, .1, 30.0)
    canvas.clear()
    assert not canvas.values().any()
    canvas.paint(np.full(78, 3.0), 2.0, 5.0, 40.0)               # too much of everything
    values = canvas.values()
    assert values.dtype == np.float32 and values.shape == (78, 3) and values.min() >= 0 and values.max() <= 1
