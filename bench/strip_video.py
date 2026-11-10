"""Render LED frames as a short video of a glowing strip, with the music, for reviewing looks.

One view (the old renders): the strip from the front. Three views, side by side: from the front; from 70 degrees
to the side, where the strip is a third as long and its near end looks larger; and the room, a field in the
strip's average light, which is most of what people at a party see of it.

Usable from both the new engine's environment and the old DJ venv: numpy, scipy and ffmpeg only."""
from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

WIDTH, HEIGHT, VIDEO_FPS = 960, 160, 30
STRIP_TOP, STRIP_HEIGHT = 56, 48
CORE_INSET = 12


def _to_rgb(frames: np.ndarray) -> np.ndarray:
    rgb = frames[..., :3].astype(np.float32)
    if frames.shape[-1] == 4:                       # the white channel lifts all three colours
        rgb = np.minimum(255, rgb + frames[..., 3:4].astype(np.float32))
    return rgb


def _picture(row: np.ndarray) -> np.ndarray:
    """One strip (n, 3) float -> (HEIGHT, WIDTH, 3) uint8: a solid core with a soft spill."""
    n = len(row)
    columns = np.minimum((np.arange(WIDTH) * n) // WIDTH, n - 1)
    line = row[columns]                              # (WIDTH, 3)
    canvas = np.zeros((HEIGHT, WIDTH, 3), np.float32)
    canvas[STRIP_TOP:STRIP_TOP + STRIP_HEIGHT] = line
    out = gaussian_filter(canvas, sigma=(9, 5, 0)) * 1.3
    core = slice(STRIP_TOP + CORE_INSET, STRIP_TOP + STRIP_HEIGHT - CORE_INSET)
    out[core] = np.maximum(np.maximum(out[core], line), 10)     # unlit strip shows as near-black, not nothing
    return np.clip(out, 0, 255).astype(np.uint8)


PANEL_W, PANEL_H = 480, 240
PANEL_MARGIN, PANEL_CORE, PANEL_GLOW = 40, 12.0, 16.0      # the strip's ends, half its thickness, its spill (pixels)
SIDE = math.cos(math.radians(70))                         # how long the strip looks from 70 degrees to the side
LIGHT = 2.2                                               # a frame's values to light and back


@lru_cache(maxsize=4)
def _layout(n: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """For the front and the side panel: per column the strip pixel it shows, and the panel's core and glow masks."""
    x = np.arange(PANEL_W)
    y = np.abs(np.arange(PANEL_H)[:, None] - (PANEL_H - 1) / 2)
    length = PANEL_W - 2 * PANEL_MARGIN
    side_length = round(length * SIDE)
    panels = []
    for start, span, depth in ((PANEL_MARGIN, length, False), ((PANEL_W - side_length) // 2, side_length, True)):
        across = np.clip((x - start) / span, 0.0, 1.0 - 1e-9)
        inside = (x >= start) & (x < start + span)
        # From the side the far end is twice as far away as the near one: it is half as thick, and the near
        # third of the strip takes half of what is seen.
        along = across / (2 - across) if depth else across
        thick = PANEL_CORE * (1.4 * (1 - across / 2) if depth else np.ones(PANEL_W))
        core = inside[None, :] & (y <= thick[None, :])
        glow = np.exp(-.5 * (np.maximum(y - thick[None, :], 0.0) / PANEL_GLOW) ** 2) * .55
        panels.append((np.minimum((along * n).astype(int), n - 1), core, np.where(inside[None, :], 1.0, .6) * glow))
    return panels


def _panels(row: np.ndarray) -> np.ndarray:
    """One strip (n, 3) float, as a screen shows it -> (PANEL_H, 3 * PANEL_W, 3) uint8: front, side, room."""
    out = np.zeros((PANEL_H, 3 * PANEL_W, 3), np.float32)
    for k, (index, core, glow) in enumerate(_layout(len(row))):
        line = np.where(core.any(axis=0)[:, None], row[index], 0.0)                   # (PANEL_W, 3)
        spill = gaussian_filter1d(line, sigma=6, axis=0)
        panel = glow[:, :, None] * spill[None, :, :]
        panel = np.where(core[:, :, None], np.maximum(np.maximum(panel, line[None, :, :]), 10), panel)
        out[:, k * PANEL_W:(k + 1) * PANEL_W] = panel
    light = (np.clip(row, 0, 255) / 255) ** LIGHT                                     # the room adds light, not values
    out[:, 2 * PANEL_W:] = light.mean(axis=0) ** (1 / LIGHT) * 255
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def render_mp4(frames: np.ndarray, fps: float, out: Path, audio: np.ndarray | None = None, rate: int = 48000,
               views: int = 1, video_fps: int = VIDEO_FPS) -> None:
    """frames: (T, n, 3|4) uint8 at `fps`. Writes H.264 (+ AAC when audio is given) at `video_fps`, with one view
    of the strip (the old renders) or three (front, side, room)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    duration = len(frames) / fps
    count = max(1, int(round(duration * video_fps)))
    picks = np.minimum((np.arange(count) * fps / video_fps).astype(int), len(frames) - 1)
    rgb = _to_rgb(frames)
    picture, size = (_panels, f"{3 * PANEL_W}x{PANEL_H}") if views == 3 else (_picture, f"{WIDTH}x{HEIGHT}")
    with tempfile.TemporaryDirectory() as tmp:
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", size,
               "-r", str(video_fps), "-i", "-"]
        if audio is not None:
            raw = Path(tmp) / "audio.f32"
            np.ascontiguousarray(audio[: int(duration * rate)], dtype="<f4").tofile(raw)
            cmd += ["-f", "f32le", "-ar", str(rate), "-ac", "2", "-i", str(raw), "-c:a", "aac", "-b:a", "128k"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-movflags", "+faststart", "-shortest", str(out)]
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for index in picks:
            process.stdin.write(picture(rgb[index]).tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {out}")
