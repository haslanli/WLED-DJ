import numpy as np
import pytest

from wled_dj.analysis.resample import Resampler


def tone(freq, seconds=.5, rate=48000):
    t = np.arange(int(seconds * rate)) / rate
    x = np.sin(2 * np.pi * freq * t).astype(np.float32)
    return np.column_stack((x, x))


def by_hops(resampler, x):
    return np.concatenate([resampler.push(x[i:i + 480]) for i in range(0, len(x), 480)])


@pytest.mark.parametrize("freq, tolerance", [(1000, 1e-3), (10000, 1e-2), (16000, 2e-2)])
def test_sine_survives(freq, tolerance):
    resampler = Resampler()
    y = by_hops(resampler, tone(freq))
    n = np.arange(len(y))
    wanted = np.sin(2 * np.pi * freq * (n / 44100 - resampler.delay_s))
    assert np.max(np.abs(y[441:, 0] - wanted[441:])) < tolerance
    assert np.array_equal(y[:, 0], y[:, 1])


def test_hops_equal_one_push():
    x = (np.random.default_rng(3).standard_normal((9600, 2)) * .2).astype(np.float32)
    assert np.allclose(by_hops(Resampler(), x), Resampler().push(x), atol=1e-6)


def test_above_the_new_nyquist_is_reduced():
    y = by_hops(Resampler(), tone(23500))
    level = 20 * np.log10(np.sqrt(np.mean(y[441:] ** 2)) / np.sqrt(.5))
    assert level < -12


def test_shape_and_delay():
    resampler = Resampler()
    y = resampler.push(np.zeros((480, 2), np.float32))
    assert y.shape == (441, 2) and y.dtype == np.float32
    assert 0 < resampler.delay_s < .0005
    with pytest.raises(ValueError):
        resampler.push(np.zeros((500, 2), np.float32))
