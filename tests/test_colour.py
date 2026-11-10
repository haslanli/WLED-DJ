"""Perceptual colour for the looks: OKLab, linear light, look values, and the palette ring."""
import numpy as np

from wled_dj.looks.colour import (RING, decode, encode, fit, linear_to_oklab, luminance, max_chroma,
                                  oklab_to_linear)


def hue_of(lab):
    return np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360


def chroma_of(lab):
    return np.hypot(lab[..., 1], lab[..., 2])


def apart(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)


def lab_of(position, lightness):
    L, C, h = RING.colour(position, lightness)
    return np.stack([np.asarray(L, float) + 0 * C, C * np.cos(np.radians(h)), C * np.sin(np.radians(h))], axis=-1)


def test_known_colours():
    red, blue, white = np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), np.ones(3)
    assert np.allclose(linear_to_oklab(red), [.628, .225, .126], atol=.001)
    assert np.allclose(linear_to_oklab(blue), [.452, -.032, -.312], atol=.001)
    assert np.allclose(linear_to_oklab(white), [1, 0, 0], atol=1e-4)
    for rgb in (red, blue, white, np.array([.2, .5, .1])):
        assert np.allclose(oklab_to_linear(linear_to_oklab(rgb)), rgb, atol=1e-4)


def test_eighteen_percent_looks_half():
    assert np.allclose(fit(.5646, 0.0, 0.0), .18, atol=.002)
    assert abs(float(luminance(fit(.5646, 0.0, 0.0))) - .18) < .002


def test_fit_keeps_lightness_and_hue():
    hues = np.arange(0, 360, 15.0)
    for lightness in (.1, .45, .9):
        rgb = fit(np.full(len(hues), lightness), np.full(len(hues), .4), hues)
        assert rgb.shape == (len(hues), 3) and rgb.min() >= 0 and rgb.max() <= 1
        lab = linear_to_oklab(rgb)
        assert np.all(np.abs(lab[:, 0] - lightness) < .01)
        assert np.all(apart(hue_of(lab), hues) < 2), apart(hue_of(lab), hues).max()
        assert np.all(chroma_of(lab) > .01)                      # it asked for more colour than there is, and got what there is
        assert np.all(chroma_of(lab) <= max_chroma(np.full(len(hues), lightness), hues) + 1e-3)


def test_values_and_light_are_inverses():
    x = np.linspace(0, 1, 101)
    assert np.allclose(decode(encode(x)), x, atol=1e-6)
    assert abs(float(encode(.18)) ** 2.2 - .18) < 1e-9
    assert float(encode(-1.0)) == 0 and float(encode(2.0)) == 1


def test_the_ring_is_a_closed_loop():
    for lightness in (.5, .15):
        assert np.allclose(lab_of(0.0, lightness), lab_of(1.0, lightness), atol=1e-9)
        walk = lab_of(np.arange(361) / 360, lightness)
        steps = np.linalg.norm(np.diff(walk, axis=0), axis=1)
        assert steps.max() < .03, steps.max()
    assert np.allclose(lab_of(1.25, .5), lab_of(.25, .5)) and np.allclose(lab_of(-.25, .5), lab_of(.75, .5))


def test_eight_anchors_are_eight_colours():
    hues = hue_of(lab_of(np.arange(8) / 8, .6))
    for i in range(8):
        for j in range(i + 1, 8):
            assert apart(hues[i], hues[j]) >= 25, (i, j, hues)
        assert apart(hues[i], hues[(i + 4) % 8]) >= 120, (i, hues)
    assert np.all(chroma_of(lab_of(np.arange(8) / 8, .6)) > .08)       # and each of them is a colour, not a grey


def test_nothing_dim_is_green():
    positions = np.arange(720) / 720
    for lightness in (.3, .2, .12, .05):                # 0.3: the first render showed 6% dim green on one clip, in the blend above 0.2
        lab = lab_of(positions, lightness)
        green = (hue_of(lab) > 110) & (hue_of(lab) < 215) & (chroma_of(lab) > .04)
        assert not green.any(), (lightness, positions[green][:5])
    seam = lab_of(11 / 16, .6)                       # between the yellow anchor and the azure one, in good light: green is fine
    assert 130 < hue_of(seam) < 200 and chroma_of(seam) > .08
    assert chroma_of(lab_of(11 / 16, .3)) < .04      # the same place when dim: no colour at all, on its way from yellow to azure
