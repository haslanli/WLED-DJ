import numpy as np

from wled_dj.compositor import Compositor
from wled_dj.flash import largest_in_a_second


def test_full_white_is_255():
    out = Compositor().compose(np.ones((78, 3), np.float32), 0.0)
    assert out.dtype == np.uint8 and out.shape == (78, 3) and (out == 255).all()


def test_master_with_linear_gamma():
    c = Compositor(gamma=1.0)
    c.master = .5
    assert (c.compose(np.ones((4, 3), np.float32), 0.0) == 128).all()


def test_master_with_gamma():
    c = Compositor(gamma=2.2)
    c.master = .5
    assert (c.compose(np.ones((4, 3), np.float32), 0.0) == round(255 * .5 ** 2.2)).all()


def test_blackout():
    c = Compositor()
    c.blackout = True
    assert not c.compose(np.ones((4, 3), np.float32), 0.0).any()


def test_fade_multiplies():
    c = Compositor(gamma=1.0)
    c.fade = .25
    assert (c.compose(np.ones((4, 3), np.float32), 0.0) == 64).all()


def test_out_of_range_input_is_clipped():
    c = Compositor(gamma=1.0)
    out = c.compose(np.array([[2.0, -1.0, .5]], np.float32), 0.0)
    assert out.tolist() == [[255, 0, 128]]


def test_gamma_can_change_at_runtime():
    c = Compositor(gamma=2.2)
    c.gamma = 1.0
    assert (c.compose(np.full((2, 3), .5, np.float32), 0.0) == 128).all()


def test_preview_is_perceptual_and_respects_controls():
    c = Compositor(gamma=2.2)
    c.master = .5
    c.compose(np.ones((2, 3), np.float32), 0.0)
    assert (c.preview() == 128).all()
    c.blackout = True
    c.compose(np.ones((2, 3), np.float32), .01)
    assert not c.preview().any()


# ---- the flash limit -------------------------------------------------------------------------------------

FPS = 80


def strobe(seconds, level=1.0):
    """(T, 78, 3) look values: dark, then the whole strip at `level`, four frames each (10 Hz)."""
    frames = np.zeros((int(seconds * FPS), 78, 3), np.float32)
    frames[np.arange(len(frames)) % 8 >= 4] = level
    return frames


def lights(compositor, frames):
    """What the LEDs get, as linear light, and what the preview shows, frame by frame."""
    sent, shown = [], []
    for i, frame in enumerate(frames):
        out = compositor.compose(frame, i / FPS) / 255
        sent.append(out if compositor.gamma == 2.2 else out ** 2.2)
        shown.append(compositor.preview().copy())
    return np.stack(sent), np.stack(shown)


def test_a_strobe_reaches_the_lights_at_three_a_second():
    for gamma in (2.2, 1.0):
        compositor = Compositor(gamma=gamma)
        assert compositor.flash_limit == 3
        sent, _ = lights(compositor, strobe(4))
        assert largest_in_a_second(sent, FPS) == 3, gamma
        assert compositor.held > 0
        compositor = Compositor(gamma=gamma)
        compositor.flash_limit = 0
        sent, _ = lights(compositor, strobe(4))
        assert largest_in_a_second(sent, FPS) == 10 and compositor.held == 0


def test_the_preview_shows_what_the_lights_show():
    compositor = Compositor()
    _, shown = lights(compositor, strobe(4))
    assert (shown[4] == 255).all()                     # the first flash, as the look drew it
    assert shown[28].max() < 100 and shown[28].min() > 0          # the fourth, held: the preview is dim too
    assert (shown[0] == 0).all()


def test_blackout_and_master_come_before_the_limit():
    compositor = Compositor()
    compositor.blackout = True
    sent, shown = lights(compositor, strobe(2))
    assert not sent.any() and not shown.any() and compositor.held == 0
    compositor = Compositor()
    compositor.master = .2                             # a fifth of full: the strobe's light rises by less than a tenth
    sent, _ = lights(compositor, strobe(2))
    assert compositor.held == 0 and largest_in_a_second(sent, FPS) == 0
    assert (compositor.compose(np.ones((78, 3), np.float32), 9.0) == round(255 * .2 ** 2.2)).all()


def test_a_strip_of_another_length_mid_run():
    compositor = Compositor()
    for i, frame in enumerate(strobe(1)):
        compositor.compose(frame, i / FPS)
    assert compositor.compose(np.ones((30, 3), np.float32), 1.0).shape == (30, 3) and compositor.preview().shape == (30, 3)
