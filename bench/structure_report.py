"""What the engine reads of structure, tone and lead on every file in bench/corpus, on stems and on the mix.

Per genre (and for the full tracks): share of time in each section, boundaries and drops a minute, how far the
song's colour wanders, and who has the focus (how often each group is the figure, and its mean weight). A file's
first 3 s are left out: the readings are still settling. A file without cached stems is separated first (it
needs the model files), so music added to bench/corpus is measured by the same command.

Usage: uv run python bench/structure_report.py [--workers 8] [--only PATTERN]
Writes bench/out/structure.json."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from replay import CACHE, analyse, cached_stems, decode
from wled_dj.analysis.structure import SECTIONS
from wled_dj.stems import protocol
from wled_dj.stems.models import models_dir, verified

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
OUT = HERE / "out" / "structure.json"
AUDIO = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg")
GROUPS = ("drums", "harmony", "lead")
SETTLING = 300                       # hops left out at the start of every file


def separate(audio: Path) -> None:
    """Run the model over the file and cache its stems, as bench/score_layers.py does for its songs."""
    from wled_dj.stems.runner import StemModel
    mix = decode(audio, protocol.RATE)
    model = StemModel(models_dir())
    hop, count = protocol.HOP, len(mix) // protocol.HOP
    est = np.zeros((count * hop, 3), np.float32)
    for k in range(count + 1):                      # the answer to chunk k is the stems of chunk k - 1
        chunk = mix[k * hop:(k + 1) * hop].T if k < count else np.zeros((2, hop), np.float32)
        out = model.step(chunk)
        if k:
            est[(k - 1) * hop:k * hop] = out.T
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / (audio.stem + ".npz"), est=np.clip(est * 32767, -32768, 32767).astype(np.int16))


def arc(angles: np.ndarray) -> float:
    """The smallest arc of the circle, in degrees, that holds all the angles."""
    if len(angles) < 2:
        return 0.0
    ordered = np.sort(np.mod(angles, 360))
    gaps = np.diff(np.concatenate((ordered, ordered[:1] + 360)))
    return float(360 - gaps.max())


def measure(features) -> dict:
    features = features[SETTLING:]
    seconds = len(features) / 100
    structure = [f.structure for f in features]
    trusted = np.array([f.tone.angle for f in features if f.tone.confidence > 0])
    figure = np.array([int(np.argmax(s.focus)) for s in structure])
    return {"seconds": round(seconds, 1),
            "section": {name: round(sum(s.section == name for s in structure) / len(structure), 3) for name in SECTIONS},
            "boundaries_per_min": round(sum(s.boundary > 0 for s in structure) / seconds * 60, 2),
            "strong_per_min": round(sum(s.boundary >= .7 for s in structure) / seconds * 60, 2),
            "drops_per_min": round(sum(s.drop for s in structure) / seconds * 60, 2),
            "energy": round(float(np.mean([s.energy for s in structure])), 3),
            "tension_high": round(sum(s.tension >= .6 for s in structure) / len(structure), 3),
            "colour_held": round(1 - len(trusted) / len(features), 3),
            "colour_arc": round(arc(trusted[len(trusted) // 4:]), 1),          # after the first quarter, once it has settled
            "figure": {name: round(float(np.mean(figure == i)), 3) for i, name in enumerate(GROUPS)},
            "focus": {name: round(float(np.mean([s.focus[i] for s in structure])), 3) for i, name in enumerate(GROUPS)}}


def one(audio: Path) -> dict:
    if cached_stems(audio) is None:
        separate(audio)
    meta = audio.with_suffix(".json")
    genre = json.loads(meta.read_text()).get("genre", "?") if meta.exists() else ("full track" if audio.parent == CORPUS else "added")
    return {"file": str(audio.relative_to(CORPUS)), "genre": genre,
            "stems": measure(analyse(audio)), "mix": measure(analyse(audio, stems=False))}


def mean(rows: list[dict], source: str, *keys: str) -> float:
    values = []
    for row in rows:
        value = row[source]
        for key in keys:
            value = value[key]
        values.append(value)
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    files = sorted(p for p in CORPUS.rglob("*") if p.suffix.lower() in AUDIO and args.only in p.name)
    if not files:
        raise SystemExit(f"no audio in {CORPUS}: see bench/DATA.md")
    missing = [p for p in files if cached_stems(p) is None]
    if missing and not verified(models_dir()):
        raise SystemExit(f"{len(missing)} files have no cached stems and there are no model files in {models_dir()}: "
                         "set WLED_DJ_MODELS (for example to bench/data/stemgenrt)")
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(one, files))
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["file"] if row["genre"] == "full track" else row["genre"], []).append(row)
    head = (f"{'':24} {'n':>3} | {'quiet':>5} {'groov':>5} {'break':>5} {'build':>5} {'drop':>5} | {'bound':>5} {'strong':>6} {'drops':>5} /min | "
            f"{'held':>4} {'arc':>4} | figure d/h/l   | weight d/h/l")
    for source in ("stems", "mix"):
        print(f"\nfrom {source}\n{head}")
        for name in sorted(groups):
            g = groups[name]
            print(f"{name[:24]:24} {len(g):3d} | " + " ".join(f"{mean(g, source, 'section', s):5.2f}" for s in SECTIONS)
                  + f" | {mean(g, source, 'boundaries_per_min'):5.1f} {mean(g, source, 'strong_per_min'):6.1f} {mean(g, source, 'drops_per_min'):5.1f}      | "
                  f"{mean(g, source, 'colour_held'):4.2f} {mean(g, source, 'colour_arc'):4.0f} | "
                  + "/".join(f"{mean(g, source, 'figure', k):.2f}" for k in GROUPS) + " | "
                  + "/".join(f"{mean(g, source, 'focus', k):.2f}" for k in GROUPS))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"files": rows}, indent=1))
    print(f"\n{len(rows)} files; written to {OUT.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
