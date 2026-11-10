"""The real stem helper. Needs the StemgenRT model files: WLED_DJ_MODELS=bench/data/stemgenrt uv run pytest -m models"""
import sys
import time

import numpy as np
import pytest

from signals import music_pattern
from wled_dj.analysis.resample import Resampler
from wled_dj.stems.link import StemLink
from wled_dj.stems.models import models_dir, verified
from wled_dj.stems.runner import StemModel

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def directory():
    if not verified(models_dir()):
        pytest.skip(f"no verified model files in {models_dir()}")
    return models_dir()


def test_real_helper_separates_and_aligns(directory):
    link = StemLink([sys.executable, "-m", "wled_dj.stems", str(directory)])
    link.start()
    try:
        deadline = time.monotonic() + 30
        while link.state != "on" and time.monotonic() < deadline:
            time.sleep(.05)
        assert link.state == "on"
        x = music_pattern(6.0)
        hops = []
        for n in range(600):
            link.push(x[n * 480:(n + 1) * 480], n * .01)
            hops += link.collect(1.0)
    finally:
        link.stop()
    assert [hop.index for hop in hops] == list(range(600))
    drums = np.concatenate([hop.drums for hop in hops[100:]])
    vocals = np.concatenate([hop.vocals for hop in hops[100:]])
    assert np.mean(drums ** 2) > 10 * np.mean(vocals ** 2)        # kicks and hats are drums, not a voice
    # The same audio straight through the model: answer k is the stems of chunk k - 1.
    model = StemModel(directory)
    audio = Resampler().push(x)
    answers = [model.step(np.ascontiguousarray(audio[k * 128:(k + 1) * 128].T)) for k in range(len(audio) // 128)]
    direct = np.concatenate((np.zeros(256, np.float32), np.concatenate([a[0] for a in answers[1:]])))
    for n in (20, 150, 400):
        assert np.allclose(hops[n].drums, direct[441 * n:441 * (n + 1)], atol=1e-4)


def test_model_step_shapes(directory):
    out = StemModel(directory).step(np.zeros((2, 128), np.float32))
    assert out.shape == (3, 128) and out.dtype == np.float32
