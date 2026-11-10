"""What a flash is (WCAG 2.3.1), the limiter that holds any picture to a number of them, and a look's budget.

Frames here come 80 a second, so a 10 Hz strobe is exactly four frames on and four off."""
import numpy as np

from wled_dj.flash import FlashBudget, FlashLimiter, FlashMeter, largest_in_a_second, luminance

FPS, PIXELS = 80, 78


def strobe(seconds, hz=10, share=1.0, colour=(1.0, 1.0, 1.0)):
    """(T, PIXELS, 3) linear light: the first `share` of the strip in `colour` for the second half of every
    period (a picture's very first frame has nothing before it to be a flash against)."""
    period = FPS // hz
    frames = np.zeros((int(seconds * FPS), PIXELS, 3))
    lit = int(round(share * PIXELS))
    for i in range(len(frames)):
        if i % period >= period // 2:
            frames[i, :lit] = colour
    return frames


def through(limiter, frames):
    return np.stack([limiter.step(frame, i / FPS) for i, frame in enumerate(frames)])


def test_a_ten_hertz_strobe_is_held_to_three():
    frames = strobe(4)
    assert largest_in_a_second(frames, FPS) == 10
    limiter = FlashLimiter(3)
    out = through(limiter, frames)
    assert largest_in_a_second(out, FPS) == 3
    assert np.array_equal(out[:28], frames[:28])                 # the first three flashes come through as they are
    assert frames[28:32].min() == 1 and out[28:32].max() < .11 and limiter.held > 0      # the fourth is held under a tenth
    assert out.max() <= 1 and out.min() >= 0


def test_a_two_hertz_pulse_passes_untouched():
    frames = strobe(4, hz=2)
    limiter = FlashLimiter(3)
    assert np.array_equal(through(limiter, frames), frames) and limiter.held == 0
    assert largest_in_a_second(frames, FPS) == 2


def test_small_flicks_pass_untouched():
    frames = strobe(4, hz=10, share=.1)
    limiter = FlashLimiter(3)
    assert np.array_equal(through(limiter, frames), frames) and limiter.held == 0
    assert largest_in_a_second(frames, FPS) == 0


def test_off_passes_everything():
    frames = strobe(4)
    limiter = FlashLimiter(0)
    assert np.array_equal(through(limiter, frames), frames) and limiter.held == 0


def test_saturated_red_counts_even_when_dim():
    red = (.05 / .2126, 0.0, 0.0)                                # luminance 0.05: under a tenth, but red
    assert abs(float(luminance(np.array(red))) - .05) < 1e-9
    frames = strobe(4, colour=red)
    assert largest_in_a_second(frames, FPS) == 10
    out = through(FlashLimiter(3), frames)
    assert largest_in_a_second(out, FPS) == 3
    grey = strobe(4, colour=(.05, .05, .05))                     # the same luminance without the red
    limiter = FlashLimiter(3)
    assert np.array_equal(through(limiter, grey), grey) and largest_in_a_second(grey, FPS) == 0


def test_a_steady_rise_is_not_a_flash_for_ever():
    frames = np.zeros((4 * FPS, PIXELS, 3))
    frames[FPS:] = .3                                            # dark, then light for good
    frames[int(2.5 * FPS):] = .5                                 # and a step more, 1.5 s later
    meter = FlashMeter()
    starts = [i / FPS for i, frame in enumerate(frames) if meter.step(frame, i / FPS)]
    assert starts == [1.0, 2.5]


def test_the_budget_refuses_the_fourth():
    budget = FlashBudget(3)
    assert [budget.ask(t) for t in (0.0, .2, .4, .6, 1.05)] == [True, True, True, False, True]
    assert all(FlashBudget(0).ask(t / 100) for t in range(50))   # no limit: every gesture may be large


def test_state_follows_the_strip():
    limiter = FlashLimiter(3)
    frames = strobe(6)
    for i, frame in enumerate(frames):
        pixels = 30 if FPS <= i < 2 * FPS else PIXELS            # a shorter strip for a second, then back
        limiter.limit = (3, 1, 0, 3, 3, 3)[i // FPS]
        out = limiter.step(frame[:pixels], i / FPS)
        assert out.shape == (pixels, 3)
        if limiter.limit == 0:
            assert np.array_equal(out, frame[:pixels])
    out = through(FlashLimiter(1), strobe(4))
    assert largest_in_a_second(out, FPS) == 1


def test_time_stepping_back_a_little_changes_nothing():
    """A change of the features' source moves their clock by a few milliseconds."""
    frames = strobe(4)
    limiter = FlashLimiter(3)
    out = np.stack([limiter.step(frame, i / FPS - (.012 if 100 <= i < 104 else 0.0)) for i, frame in enumerate(frames)])
    assert largest_in_a_second(out, FPS) == 3


def test_nothing_grows():
    meter, limiter, budget = FlashMeter(), FlashLimiter(3), FlashBudget(3)
    frames = strobe(1)
    for i in range(60 * FPS):
        frame = frames[i % FPS]
        meter.step(frame, i / FPS)
        limiter.step(frame, i / FPS)
        budget.ask(i / FPS)
    assert len(meter.starts) <= 11 and len(limiter.meter.starts) <= 3 and len(budget.granted) <= 3
    assert meter.window.light.shape[0] <= 512
