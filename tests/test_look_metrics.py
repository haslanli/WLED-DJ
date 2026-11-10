"""Numbers read from a look's rendered frames (bench/look_metrics.py), on made-up frames."""
from pathlib import Path
import shutil
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

from look_metrics import measure, picture, strip_of_video  # noqa: E402
from strip_video import render_mp4  # noqa: E402
from wled_dj.looks.colour import encode, fit  # noqa: E402

FPS = 80


def screen(lightness, chroma, hue, frames=160, pixels=78):
    """(frames, pixels, 3) uint8: every pixel this colour, as a screen shows it."""
    row = np.round(encode(fit(lightness, chroma, hue)) * 255).astype(np.uint8)
    return np.tile(row, (frames, pixels, 1))


def test_a_strobe_measures_ten():
    frames = np.zeros((4 * FPS, 78, 3), np.uint8)
    frames[np.arange(len(frames)) % 8 >= 4] = 255
    numbers = measure(frames, FPS)
    assert numbers["large_flashes"] == 10 and numbers["blink_hz"] == 10
    assert abs(numbers["dark"] - .5) < .01 and abs(numbers["lit"] - .5) < .01


def test_a_steady_violet_strip():
    numbers = measure(screen(.3, .12, 300.0), FPS)
    assert abs(numbers["lightness"] - .3) < .02
    assert numbers["dark"] == 0 and numbers["lit"] == 1 and numbers["dim_green"] == 0
    assert numbers["large_flashes"] == 0 and numbers["blink_hz"] == 0 and numbers["follows"] is None


def test_dim_green_is_found():
    assert measure(screen(.2, .08, 150.0), FPS)["dim_green"] == 1
    assert measure(screen(.6, .12, 150.0), FPS)["dim_green"] == 0           # a green in good light is fine
    assert measure(screen(.2, .01, 150.0), FPS)["dim_green"] == 0           # and a grey is not a green
    half = screen(.2, .08, 150.0)
    half[:, :39] = 0
    assert abs(measure(half, FPS)["dim_green"] - .5) < .01


def test_light_that_follows_the_music():
    t = np.arange(8 * FPS) / FPS
    energy = .5 + .4 * np.sin(2 * np.pi * t / 3)
    frames = np.round(np.clip(energy, 0, 1)[:, None, None] * np.ones((1, 78, 3)) * 255).astype(np.uint8)
    numbers = measure(frames, FPS, energy)
    assert numbers["follows"] > .95 and abs(numbers["energy_span"] - .79) < .05     # a sine's 5th to 95th percentile
    other = .5 + .4 * np.sin(2 * np.pi * t / .7 + 1)
    assert abs(measure(frames, FPS, other)["follows"]) < .3
    assert measure(screen(.3, .1, 300.0, frames=len(t)), FPS, energy)["follows"] == 0      # a strip that never changes follows nothing


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_the_strip_is_read_back_from_an_old_video(tmp_path):
    frames = np.zeros((60, 78, 3), np.uint8)
    frames[:, :26], frames[:, 26:52], frames[:, 52:] = (220, 30, 30), (0, 0, 0), (30, 60, 220)
    out = tmp_path / "old.mp4"
    render_mp4(frames, 30, out)
    read, fps = strip_of_video(out)
    assert fps == 30 and read.shape[1:] == (96, 3) and abs(len(read) - 60) <= 2
    thirds = [read[10:50, a:b].reshape(-1, 3).mean(axis=0) for a, b in ((2, 30), (34, 62), (66, 94))]
    assert np.abs(thirds[0] - (220, 30, 30)).max() <= 12 and np.abs(thirds[2] - (30, 60, 220)).max() <= 12
    assert thirds[1].max() == 0                                             # the video's dark grey strip body is no light


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_the_picture_of_a_clip(tmp_path):
    frames = np.zeros((3 * FPS, 78, 3), np.uint8)
    frames[FPS:2 * FPS] = (200, 40, 160)
    out = tmp_path / "clip.png"
    picture(frames, FPS, {"kick": [.5, 1.0, 1.5], "drop": [2.0]}, out)
    data = out.read_bytes()
    width, height = struct.unpack(">II", data[16:24])
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and width == 60 and height == 156 + 2 * 10 + 64
