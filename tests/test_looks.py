"""The look registry: one calibration look built in, and yours loaded from a folder."""
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from wled_dj.analysis.features import LEVELS, Features
from wled_dj.analysis.onsets import Onset
from wled_dj.analysis.rhythm import BeatState
from wled_dj.looks import DEFAULT, LOOKS, load_looks
from wled_dj.looks.sync_check import SyncCheck

STEP = .01
EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "looks"


def features(t, events=()):
    levels = MappingProxyType({name: 0.0 for name in LEVELS})
    kwargs = {name: Onset(name, strength, pan) for name, strength, pan in events}
    return Features(t=t, levels=levels, beat=BeatState(bpm=120.0, phase=(t * 2) % 1.0, bar_phase=0.0, confidence=1.0, t=t), **kwargs)


def play(look, until, events_at=None):
    """Render every 10 ms up to `until`; events_at maps time -> events. Returns {t: frame}."""
    events_at = events_at or {}
    return {round(i * STEP, 6): look.render(features(round(i * STEP, 6), events_at.get(round(i * STEP, 6), ())), 0.0)
            for i in range(int(round(until / STEP)) + 1)}


@pytest.fixture
def registry():
    before = dict(LOOKS)
    yield LOOKS
    LOOKS.clear()
    LOOKS.update(before)


def test_no_effects_out_of_the_box():
    assert list(LOOKS) == ["sync-check"] and DEFAULT == "sync-check" and LOOKS[DEFAULT] is SyncCheck


@pytest.mark.parametrize("n", [78, 300])
def test_shape_dtype_range(n):
    for frame in play(SyncCheck(n), 3.0, {1.0: [("kick", 1.0, 0.0)]}).values():
        assert frame.shape == (n, 3) and frame.dtype == np.float32 and frame.min() >= 0 and frame.max() <= 1


def test_sync_check_flashes_only_on_kick():
    frames = play(SyncCheck(78), 1.2, {1.0: [("kick", .5, 0.0)], 1.1: [("snare", 1.0, 0.0)]})
    assert all(frames[round(1.0 + d, 2)].min() >= .9 for d in (0, .01, .02, .03, .04, .05))
    assert frames[1.08].max() == 0 and frames[1.1].max() == 0 and frames[0.5].max() == 0


def test_the_example_look_loads_and_renders(registry):
    assert load_looks(EXAMPLES) == ["glow"]
    assert LOOKS["glow"].name == "Glow"
    look = LOOKS["glow"](78)
    frame = look.render(Features.silence(0.0), 0.0)
    assert frame.shape == (78, 3) and frame.dtype == np.float32 and not frame.any()


def test_a_folder_of_looks(registry, tmp_path, caplog):
    (tmp_path / "two.py").write_text(
        "import numpy as np\n"
        "class Red:\n    id, name = 'red', 'Red'\n    def __init__(self, pixels): self.pixels = pixels\n"
        "    def render(self, f, lead): return np.tile(np.float32([1, 0, 0]), (self.pixels, 1))\n"
        "class Blue(Red):\n    id, name = 'blue', 'Blue'\n"
        "class Helper:\n    pass\n")
    (tmp_path / "broken.py").write_text("this is not python\n")
    (tmp_path / "taken.py").write_text("class Again:\n    id, name = 'sync-check', 'Mine'\n    def render(self, f, lead): pass\n")
    (tmp_path / "notes.txt").write_text("not a look")
    assert sorted(load_looks(tmp_path)) == ["blue", "red"]
    assert set(LOOKS) == {"sync-check", "red", "blue"} and LOOKS["sync-check"] is SyncCheck
    assert LOOKS["red"](5).render(None, 0.0).shape == (5, 3)
    assert "broken.py" in caplog.text and "taken" in caplog.text
    assert load_looks(tmp_path / "missing") == []
