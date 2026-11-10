"""Numbers read from a look's rendered frames, so a render can be judged against the rules it was built on.

    measure(frames, fps, energy)      large flashes, blinking, lightness, what is lit, dim green, how the light follows the music
    strip_of_video(path)              the same frames read back from an old single-view review video
    picture(frames, fps, marks, out)  the whole clip as one still: time left to right, the strip top to bottom

Frames are (T, pixels, 3) uint8 as a screen shows them (the compositor's preview)."""
from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from strip_video import STRIP_TOP, WIDTH, _picture
from wled_dj.flash import largest_in_a_second
from wled_dj.looks.colour import decode, linear_to_oklab

LIT = .03                            # lightness from which a pixel counts as lit
BLINK, BLINK_BACK_S = .1, .2         # a blink: lightness this far above its lowest of the last 0.2 s
DIM, GREEN, COLOURED = .3, (110.0, 215.0), .04     # dim green: under this lightness, between these hues, with this much chroma
SMOOTH_S = .2
OLD_COLUMNS, OLD_NOISE = 96, 6       # an old video's strip is read as 96 pixels; levels this low are the encoder's noise
# An old video draws the strip's core brighter than the frame was (its glow is laid over it), so the strip is
# read from the six rows at the core's upper edge, where the picture is the frame times a known factor.
OLD_ROWS = np.arange(STRIP_TOP - 2, STRIP_TOP + 4)
OLD_GAIN = _picture(np.full((8, 3), 255.0, np.float32))[OLD_ROWS, WIDTH // 2, 0] / 255.0
PIXELS_A_SECOND, STRIP_ROWS, MARK_ROWS, CURVE_ROWS = 20, 156, 10, 64


def _blink_hz(lightness: np.ndarray, fps: float) -> int:
    """The most rises of BLINK in lightness that any one pixel makes inside one second."""
    back = max(1, int(round(BLINK_BACK_S * fps)))
    low = minimum_filter1d(lightness, size=back + 1, axis=0, origin=back // 2, mode="nearest")     # the lowest of now and before
    up = lightness - low >= BLINK
    starts = (up[1:] & ~up[:-1]).astype(np.int32)
    if not len(starts):
        return 0
    second = max(1, int(round(fps)))
    sums = np.cumsum(np.vstack((np.zeros((1, starts.shape[1]), np.int32), starts)), axis=0)
    if len(starts) <= second:
        return int(sums[-1].max())
    return int((sums[second:] - sums[:-second]).max())


def measure(frames: np.ndarray, fps: float, energy: np.ndarray | None = None) -> dict:
    """energy: the song's energy (0-1) at each frame, for `follows`."""
    light = decode(frames.astype(np.float64) / 255)
    lab = linear_to_oklab(light)
    lightness, chroma = lab[..., 0], np.hypot(lab[..., 1], lab[..., 2])
    hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360
    lit = lightness > LIT
    green = lit & (lightness < DIM) & (hue > GREEN[0]) & (hue < GREEN[1]) & (chroma > COLOURED)
    numbers = {"large_flashes": largest_in_a_second(light, fps), "blink_hz": _blink_hz(lightness, fps),
               "lightness": round(float(lightness.mean()), 4), "dark": round(float(1 - lit.mean()), 4),
               "lit": round(float(lit.mean(axis=1).mean()), 4), "dim_green": round(float(green.mean()), 4),
               "follows": None, "energy_span": None}
    if energy is not None:
        size = max(1, int(round(SMOOTH_S * fps)))
        total = uniform_filter1d(light.mean(axis=(1, 2)), size)
        song = uniform_filter1d(np.asarray(energy, np.float64)[:len(total)], size)
        steady = total.std() < 1e-9 or song.std() < 1e-9
        numbers["follows"] = 0.0 if steady else round(float(np.corrcoef(total[:len(song)], song)[0, 1]), 3)
        numbers["energy_span"] = round(float(np.percentile(energy, 95) - np.percentile(energy, 5)), 3)
    return numbers


def strip_of_video(path: Path) -> tuple[np.ndarray, float]:
    """((T, 96, 3) uint8, frames a second) from a review video drawn with one view of the strip."""
    probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                     "stream=r_frame_rate", "-of", "csv=p=0", str(path)], text=True).strip()
    up, down = probe.split("/")
    crop = f"crop={WIDTH}:{len(OLD_ROWS)}:0:{OLD_ROWS[0]}"
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-vf", crop, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    band = np.frombuffer(raw, np.uint8).reshape(-1, len(OLD_ROWS), WIDTH, 3) / OLD_GAIN[None, :, None, None]
    strip = band.mean(axis=1).reshape(len(band), OLD_COLUMNS, WIDTH // OLD_COLUMNS, 3).mean(axis=2)
    strip[strip.max(axis=-1) <= OLD_NOISE] = 0
    return np.clip(np.round(strip), 0, 255).astype(np.uint8), float(up) / float(down)


def picture(frames: np.ndarray, fps: float, marks: dict[str, list[float]], out: Path,
            pixels_a_second: int = PIXELS_A_SECOND) -> None:
    """A PNG of the clip: the strip over time (156 rows), under it one row of ticks per kind of mark, in the
    order given, and under those the mean lightness as a line (0 at the bottom, 1 at the top)."""
    seconds = len(frames) / fps
    width = max(1, int(round(seconds * pixels_a_second)))
    picks = np.minimum((np.arange(width) / pixels_a_second * fps).astype(int), len(frames) - 1)
    columns = frames[picks]                                                    # (width, pixels, 3)
    rows = np.minimum((np.arange(STRIP_ROWS) * frames.shape[1]) // STRIP_ROWS, frames.shape[1] - 1)
    image = np.zeros((STRIP_ROWS + MARK_ROWS * len(marks) + CURVE_ROWS, width, 3), np.uint8)
    image[:STRIP_ROWS] = columns[:, rows].transpose(1, 0, 2)
    for k, times in enumerate(marks.values()):
        top = STRIP_ROWS + k * MARK_ROWS
        image[top] = 40                                                        # a faint line between the rows
        for t in times:
            x = int(t * pixels_a_second)
            if 0 <= x < width:
                image[top + 2:top + MARK_ROWS, x] = 255
    lightness = linear_to_oklab(decode(columns.astype(np.float64) / 255))[..., 0].mean(axis=1)
    base = STRIP_ROWS + MARK_ROWS * len(marks)
    image[base] = 40
    y = base + CURVE_ROWS - 1 - np.clip(np.round(lightness * (CURVE_ROWS - 1)), 0, CURVE_ROWS - 1).astype(int)
    image[y, np.arange(width)] = 255
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{image.shape[0]}",
                    "-i", "-", "-frames:v", "1", str(out)], input=image.tobytes(), check=True)
