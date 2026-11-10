"""The review video's pictures: the old single view, and front, side and room side by side."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

from strip_video import _panels, _picture, render_mp4  # noqa: E402

PANEL = 480


def test_one_view_is_unchanged():
    row = np.stack([np.linspace(0, 255, 78), np.linspace(255, 0, 78), np.full(78, 40.0)], axis=1).astype(np.float32)
    assert hashlib.sha256(_picture(row).tobytes()).hexdigest() == "50fca238ad2a4c19227c001d1051c6b824d0ed9ae91d06b2104f11a770cf2a97"


def lit_columns(panel):
    return np.flatnonzero(panel.max(axis=(0, 2)) > 128)


def test_three_panels():
    row = np.zeros((78, 3), np.float32)
    row[:39] = (255, 0, 0)                                       # red on the left half, dark on the right
    picture = _panels(row)
    assert picture.shape == (240, 3 * PANEL, 3) and picture.dtype == np.uint8
    front, side, room = picture[:, :PANEL], picture[:, PANEL:2 * PANEL], picture[:, 2 * PANEL:]
    assert front[:, :PANEL // 2, 0].sum() > 5 * front[:, PANEL // 2:, 0].sum()        # front: left is the red half
    assert front[..., 1].max() < 40 and front[..., 2].max() < 40
    # the room: one colour, the strip's light averaged as light (half the strip at full red is 73% of full, not 50%)
    assert np.ptp(room.reshape(-1, 3), axis=0).max() <= 2
    assert abs(int(room[0, 0, 0]) - round(255 * .5 ** (1 / 2.2))) <= 10 and room[0, 0, 1] < 10
    white = _panels(np.full((78, 3), 255.0, np.float32))
    front_lit, side_lit = lit_columns(white[:, :PANEL]), lit_columns(white[:, PANEL:2 * PANEL])
    assert .25 <= len(side_lit) / len(front_lit) <= .45          # from 70 degrees to the side the strip is a third as long
    rows = lambda panel, column: int((panel[:, column].max(axis=1) > 128).sum())     # noqa: E731
    side = white[:, PANEL:2 * PANEL]
    assert rows(side, side_lit[2]) > rows(side, side_lit[-3])    # and its near end looks larger than its far end
    assert not _panels(np.zeros((78, 3), np.float32))[:, 2 * PANEL:].any()            # a dark strip lights no room


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg")
def test_a_video_is_written(tmp_path):
    frames = np.zeros((77, 78, 3), np.uint8)
    frames[::2] = 200
    out = tmp_path / "three.mp4"
    render_mp4(frames, 77, out, views=3, video_fps=60)
    stream = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                                 "stream=width,height,r_frame_rate,nb_frames", "-of", "json", str(out)]))["streams"][0]
    assert (stream["width"], stream["height"], stream["r_frame_rate"]) == (1440, 240, "60/1")
    assert 58 <= int(stream["nb_frames"]) <= 62
    old = tmp_path / "one.mp4"
    render_mp4(frames, 77, old)
    stream = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                                 "stream=width,height,r_frame_rate", "-of", "json", str(old)]))["streams"][0]
    assert (stream["width"], stream["height"], stream["r_frame_rate"]) == (960, 160, "30/1")
