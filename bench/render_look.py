"""A look on your own music, driven by the engine's own analysis of each file (bench/replay.py): from separated
stems or from the mix, whole or one layer at a time.

    uv run python bench/render_look.py --look glow --looks examples/looks song.wav other.mp3
    uv run python bench/render_look.py --look glow --looks examples/looks --start 35 --seconds 60 song.wav
    uv run python bench/render_look.py --look glow --looks examples/looks --source mix song.wav
    uv run python bench/render_look.py --look glow --looks examples/looks --layers song.wav

Per file it writes into bench/out/looks: a video (three views of the strip, 60 frames a second, with the sound),
a still of the whole clip (the strip over time, with rows of ticks for kicks, snares, bass notes, boundaries,
drops and voice entrances, and the mean lightness as a line), and its numbers (bench/look_metrics.py) into
looks.json, which the studio's /looks page reads.

The analysis runs from the file's start to the clip's end, so the trackers and the look have heard the song;
the frames kept are the clip's. A file without cached stems is separated first (that needs the model files:
set WLED_DJ_MODELS, or run the app once so it downloads them)."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from look_metrics import measure, picture  # noqa: E402
from replay import LAYERS, analyse, cached_stems, decode, solo  # noqa: E402
from strip_video import render_mp4  # noqa: E402

from wled_dj.analysis.features import Features  # noqa: E402
from wled_dj.compositor import Compositor  # noqa: E402
from wled_dj.flash import largest_in_a_second  # noqa: E402
from wled_dj.looks import LOOKS, load_looks  # noqa: E402
from wled_dj.looks.colour import decode as light  # noqa: E402

OUT = HERE / "out" / "looks"
PIXELS, FPS, RATE = 78, 77, 48000
MARKS = ("kick", "snare", "bass", "boundary", "drop", "voice in")


def render(look_id: str, features: list[Features], pixels: int = PIXELS, fps: float = FPS, flash_limit: int = 3,
           start: float = 0.0) -> tuple[np.ndarray, int]:
    """The look and the real compositor over the features. Returns the frames from `start` on, `fps` a second, as
    a screen shows them ((T, pixels, 3) uint8), and how many frames the flash limit had to dim. flash_limit 0
    lifts the compositor's limit, to see what the look does by itself."""
    look, compositor = LOOKS[look_id](pixels), Compositor(gamma=1.0, flash_limit=flash_limit)
    frames, latest = [], np.zeros((pixels, 3), np.uint8)
    for f in features:
        compositor.compose(look.render(f, 0.0), f.t)
        latest = compositor.preview()
        while start + len(frames) / fps <= f.t:
            frames.append(latest)
    return np.stack(frames), compositor.held


def merge(index: Path, clips: list[dict], key: str, entry: dict) -> None:
    """Put a look's (new) videos, numbers and one-layer files into looks.json, keeping everything else in it."""
    index = Path(index)
    data = json.loads(index.read_text()) if index.exists() else {"clips": [], "looks": {}}
    data["clips"] = clips
    look = data["looks"].setdefault(key, {"videos": {}, "numbers": {}})
    for field, value in entry.items():
        if field == "layers":
            for clip, files in value.items():
                look.setdefault("layers", {}).setdefault(clip, {}).update(files)
        elif isinstance(value, dict):
            look.setdefault(field, {}).update(value)
        else:
            look[field] = value
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(data, indent=1, ensure_ascii=False))


def file_name(key: str, clip_id: str, layer: str | None = None) -> str:
    """A render's file name, without its ending."""
    return f"{key}--{clip_id}" + (f"--{layer}" if layer else "")


def one(job: tuple[str, str | None, str, dict, str | None, str]) -> tuple[str, str | None, str, dict]:
    """Render one clip. Returns (clip id, layer, the video's file name, its numbers)."""
    look_id, looks, source, clip, layer, out = job
    if looks:
        load_looks(Path(looks))                           # each worker is its own process
    audio, start, end = Path(clip["file"]), float(clip["start"]), float(clip["start"]) + float(clip["duration"])
    if source == "stems" and cached_stems(audio) is None:
        from structure_report import separate
        separate(audio)
    features = analyse(audio, stems=source == "stems", seconds=end, layer=layer)
    frames, held = render(look_id, features, start=start)
    alone, _ = render(look_id, features, flash_limit=0, start=start)
    inside = [f for f in features if start <= f.t < end]
    energy = np.interp(start + np.arange(len(frames)) / FPS, [f.t for f in features], [f.structure.energy for f in features])
    numbers = measure(frames, FPS, energy)
    numbers.update(held=held, unlimited_flashes=largest_in_a_second(light(alone / 255), FPS))
    name = file_name(look_id + ("-mix" if source == "mix" else ""), clip["id"], layer)
    sound = solo(audio, layer) if layer else decode(audio, RATE)
    render_mp4(frames, FPS, Path(out) / f"{name}.mp4", audio=sound[int(start * RATE):int(end * RATE)], rate=RATE, views=3, video_fps=60)
    seen = {"kick": [f.t for f in inside if f.kick], "snare": [f.t for f in inside if f.snare],
            "bass": [f.t for f in inside if f.bass],
            "boundary": [f.t for f in inside if f.structure.boundary > 0], "drop": [f.t for f in inside if f.structure.drop],
            "voice in": [f.t for f in inside if f.structure.vocal_entry]}
    picture(frames, FPS, {kind: [t - start for t in seen[kind]] for kind in MARKS}, Path(out) / f"{name}.png")
    return clip["id"], layer, f"{name}.mp4", numbers


def clip_of(audio: Path, start: float, seconds: float | None) -> dict:
    """What to render of a file: all of it from `start`, or `seconds` of it."""
    length = len(decode(audio, 8000)) / 8000
    return {"id": re.sub(r"[^A-Za-z0-9_-]+", "-", audio.stem).strip("-") or "clip", "title": audio.stem, "group": "clips",
            "file": str(audio), "start": start, "duration": min(seconds or length, length - start)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--look", required=True)
    parser.add_argument("--looks", type=Path, help="a folder of look files to load first")
    parser.add_argument("--source", default="stems", choices=("stems", "mix"))
    parser.add_argument("--layers", action="store_true", help="render each file one layer at a time instead")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.looks:
        load_looks(args.looks)
    if args.look not in LOOKS:
        raise SystemExit(f"no look {args.look!r}: there are {', '.join(LOOKS)}; --looks DIR loads more")
    if args.layers and args.source != "stems":
        raise SystemExit("one layer at a time needs the stems")
    clips = [clip_of(audio.resolve(), args.start, args.seconds) for audio in args.audio]
    index = args.out / "looks.json"
    known = json.loads(index.read_text())["clips"] if index.exists() else []
    shown = known + [{"id": c["id"], "title": c["title"], "group": c["group"]} for c in clips if c["id"] not in {k["id"] for k in known}]
    jobs = [(args.look, str(args.looks) if args.looks else None, args.source, clip, layer, str(args.out))
            for clip in clips for layer in (LAYERS if args.layers else (None,))]
    key = args.look + ("-mix" if args.source == "mix" else "")
    entry: dict = {"name": LOOKS[args.look].name + (" (from the mix)" if args.source == "mix" else ""), "kind": "new"}
    with ProcessPoolExecutor(args.workers) as pool:
        for clip_id, layer, video, numbers in pool.map(one, jobs):
            if layer:
                entry.setdefault("layers", {}).setdefault(clip_id, {})[layer] = video
                entry.setdefault("layer_numbers", {})[f"{clip_id}--{layer}"] = numbers
            else:
                entry.setdefault("videos", {})[clip_id] = video
                entry.setdefault("numbers", {})[clip_id] = numbers
            print(video, numbers, flush=True)
    merge(index, shown, key, entry)


if __name__ == "__main__":
    main()
