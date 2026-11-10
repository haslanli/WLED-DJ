"""Rendering a look from analysed features (bench/render_look.py), on made-up audio and a made-up stems cache."""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import replay  # noqa: E402
from render_look import clip_of, file_name, merge, render  # noqa: E402
from signals import RATE, kick, music_pattern, sine  # noqa: E402
from wled_dj.analysis.analyzer import Analyzer  # noqa: E402
from wled_dj.capture import AudioBlock  # noqa: E402
from wled_dj.looks import LOOKS, load_looks  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "looks"


@pytest.fixture(autouse=True)
def with_the_example_look():
    before = dict(LOOKS)
    load_looks(EXAMPLES)
    yield
    LOOKS.clear()
    LOOKS.update(before)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def features_of(audio):
    analyzer, out = Analyzer(), []
    for start in range(0, len(audio) - 511, 512):
        out.extend(analyzer.feed(AudioBlock(audio[start:start + 512], start / RATE, 0), 0.0))
    return out


def test_render_gives_screen_frames():
    features = features_of(music_pattern(6.0))
    frames, held = render("glow", features)
    assert frames.dtype == np.uint8 and frames.shape[1:] == (78, 3) and abs(len(frames) - 6 * 77) <= 3
    assert held == 0 and frames[len(frames) // 2:].any()
    later, _ = render("glow", features, start=4.0)                 # a clip: the look has seen the song, the frames are the clip's
    assert abs(len(later) - 2 * 77) <= 3 and np.array_equal(later[-50:], frames[-50:])
    short, _ = render("sync-check", features, pixels=30)
    assert short.shape[1:] == (30, 3) and short.any()


def test_the_limit_can_be_lifted_to_see_the_look_alone():
    class Strobe:
        id, name = "strobe", "Strobe"

        def __init__(self, pixels):
            self.pixels = pixels

        def render(self, f, lead):
            return np.full((self.pixels, 3), float(int(f.t * 20) % 2), np.float32)      # 10 Hz, the whole strip

    from wled_dj.flash import largest_in_a_second
    from wled_dj.looks.colour import decode
    LOOKS["strobe"] = Strobe
    try:
        features = features_of(music_pattern(4.0))
        limited, held = render("strobe", features)
        free, none_held = render("strobe", features, flash_limit=0)
    finally:
        del LOOKS["strobe"]
    assert held > 0 and none_held == 0
    assert largest_in_a_second(decode(limited / 255), 77) <= 3 < largest_in_a_second(decode(free / 255), 77)


@needs_ffmpeg
def test_a_solo_layer_is_heard_alone(tmp_path, monkeypatch):
    seconds = 5.0
    beats = np.arange(0, seconds, 60 / 128)
    drums = kick(beats, seconds, rate=44100)[:, 0]
    voice = sine(440.0, seconds, .2, rate=44100)[:, 0]
    n = len(drums)
    song = tmp_path / "song.wav"
    wavfile.write(song, 44100, np.round(np.column_stack((drums + voice, drums + voice)) * 32767).astype(np.int16))
    monkeypatch.setattr(replay, "CACHE", tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    est = np.stack((drums, np.zeros(n), voice), axis=1)
    np.savez_compressed(tmp_path / "cache" / "song.npz", est=np.round(est * 32767).astype(np.int16))
    whole = replay.analyse(song)
    assert whole[-1].layers.vocals.present and whole[-1].layers.drums.present
    alone = replay.analyse(song, layer="drums")
    assert not any(f.layers.vocals.present for f in alone[150:]) and alone[-1].layers.drums.present
    assert sum(f.kick is not None for f in alone) >= 6
    sung = replay.analyse(song, layer="vocals")
    assert sung[-1].layers.vocals.present and not any(f.layers.drums.present for f in sung[150:])
    sound = replay.solo(song, "drums")
    assert sound.shape[1] == 2 and abs(len(sound) - n * 48000 / 44100) <= .01 * n
    assert np.abs(sound).max() > .2 and np.abs(replay.solo(song, "bass")).max() < 1e-3


def test_the_index_is_merged_not_replaced(tmp_path):
    index = tmp_path / "looks.json"
    clips = [{"id": "one", "title": "One", "group": "reference"}]
    merge(index, clips, "glow", {"name": "Glow", "kind": "new", "videos": {"one": "glow--one.mp4"}, "numbers": {"one": {"lit": .5}}})
    merge(index, clips, "other", {"name": "Other", "kind": "new", "videos": {"one": "other--one.mp4"}, "numbers": {"one": {"lit": .9}}})
    merge(index, clips, "glow", {"videos": {"two": "glow--two.mp4"}, "numbers": {"two": {"lit": .4}}})        # one more clip of a look that is there
    merge(index, clips, "glow", {"layers": {"one": {"drums": "glow--one--drums.mp4"}}})
    data = json.loads(index.read_text())
    assert data["clips"] == clips and set(data["looks"]) == {"glow", "other"}
    glow = data["looks"]["glow"]
    assert glow["name"] == "Glow" and glow["videos"] == {"one": "glow--one.mp4", "two": "glow--two.mp4"}
    assert glow["numbers"]["one"] == {"lit": .5} and glow["numbers"]["two"] == {"lit": .4}
    assert glow["layers"] == {"one": {"drums": "glow--one--drums.mp4"}}
    merge(index, clips, "glow", {"videos": {"one": "glow--one-b.mp4"}, "numbers": {"one": {"lit": .6}}})      # rendered again
    assert json.loads(index.read_text())["looks"]["glow"]["videos"]["one"] == "glow--one-b.mp4"
    assert json.loads(index.read_text())["looks"]["other"]["numbers"]["one"] == {"lit": .9}


def test_file_names_and_clips(tmp_path):
    assert file_name("glow", "song") == "glow--song" and file_name("glow-mix", "song", "drums") == "glow-mix--song--drums"
    song = tmp_path / "My song (live).wav"
    wavfile.write(song, 44100, np.zeros((44100 * 3, 2), np.int16))
    whole = clip_of(song, 0.0, None)
    assert whole["id"] == "My-song-live" and whole["title"] == "My song (live)" and abs(whole["duration"] - 3.0) < .01
    part = clip_of(song, 1.0, 60.0)
    assert part["start"] == 1.0 and abs(part["duration"] - 2.0) < .01
