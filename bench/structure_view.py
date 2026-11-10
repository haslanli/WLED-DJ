"""A page to watch structure, tone and lead on: per clip a player and, under it, one timeline with a moving
playhead, drawn from the engine's own analysis of the file (bench/replay.py).

    uv run python bench/structure_view.py [clip ...]      # default: the full tracks and one clip per genre
    uv run python bench/structure_view.py --serve 8774     # show the page that is there, on this computer only

A clip is a path, or a part of a file name in bench/corpus. Writes bench/out/structure/index.html. Plain HTML,
inline SVG and script, nothing loaded from the internet; the audio is referenced in bench/corpus by relative
path. Open the file from disk, or serve it (--serve) where files cannot be opened, as in the app's own browser."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from html import escape
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import sys
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from replay import analyse, cached_stems  # noqa: E402

from wled_dj.analysis.features import Features  # noqa: E402
from wled_dj.analysis.lead import Voice  # noqa: E402
from wled_dj.analysis.structure import SECTIONS, Structure  # noqa: E402
from wled_dj.analysis.tone import Tone  # noqa: E402

CORPUS = HERE / "corpus"
OUT = HERE / "out" / "structure" / "index.html"
AUDIO = {".wav", ".mp3", ".m4a", ".flac", ".aiff", ".ogg"}
STEP = 10                           # hops per drawn sample: curves and the readout have a value every 0.1 s
STRIP = 50                          # hops per patch of a colour strip
HEARD = .1                          # the lead's line is drawn while voice.level is over this
LAYERS = ("drums", "bass", "vocals", "other")
GROUPS = ("drums", "harmony", "lead")


def clock(t: float) -> str:
    t = max(0.0, round(t, 1))
    return f"{int(t // 60)}:{t % 60:04.1f}"


def _runs(values: list) -> list[tuple[int, int, object]]:
    """(first index, index after the last, value) for every stretch of equal values."""
    runs, start = [], 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            runs.append((start, i, values[start]))
            start = i
    return runs


def _svg(inner: str, seconds: float) -> str:
    return f'<svg viewBox="0 0 {seconds:.2f} 1" preserveAspectRatio="none" aria-hidden="true">{inner}</svg>'


def _area(ts: list[float], low: list[float], high: list[float], css: str) -> str:
    """A band between two curves of 0-1 values (0 is the lane's bottom)."""
    there = [f"{t:.2f},{1 - v:.3f}" for t, v in zip(ts, high)]
    back = [f"{t:.2f},{1 - v:.3f}" for t, v in zip(reversed(ts), reversed(low))]
    return f'<polygon class="{css}" points="{" ".join(there + back)}"/>'


def _hue(angles: list[float]) -> float:
    """The mean direction of some angles, in degrees."""
    x = sum(math.cos(math.radians(a)) for a in angles)
    y = sum(math.sin(math.radians(a)) for a in angles)
    return math.degrees(math.atan2(y, x)) % 360


def _strip(ts: list[float], tones: list[Tone], pick: str, seconds: float) -> str:
    """Patches of half a second in the hue of the tone's angle; faint while the colour is only being held."""
    patches = []
    for start in range(0, len(tones), STRIP):
        part = tones[start:start + STRIP]
        left = ts[start] - .01
        right = ts[min(start + STRIP, len(ts)) - 1]
        held = sum(tone.confidence <= 0 for tone in part) > len(part) / 2
        hue = _hue([getattr(tone, pick) for tone in part])
        patches.append(f'<rect x="{left:.2f}" y="0" width="{right - left:.2f}" height="1" fill="hsl({hue:.0f} 70% 50%)"'
                       + (' opacity=".2"' if held else "") + "/>")
    return _svg("".join(patches), seconds).replace("<svg ", '<svg shape-rendering="crispEdges" ', 1)


def timeline(features: list[Features], title: str, audio_href: str) -> str:
    """One clip: its player and its timeline, as a <section>. `features` are the analyzer's, one per 10 ms."""
    seconds = features[-1].t if features else 1.0
    ts = [f.t for f in features]
    structure = [f.structure or Structure() for f in features]
    tones = [f.tone or Tone() for f in features]
    voices = [f.voice or Voice() for f in features]
    drawn = range(0, len(features), STEP)
    xs = [ts[i] for i in drawn]

    def left(t: float) -> str:
        return f"left:{t / seconds * 100:.3f}%"

    ticks = "".join(f'<span class="{"major" if t % (60 if seconds > 150 else 30) == 0 else "minor"}" style="{left(t)}">'
                    f"{t // 60}:{t % 60:02d}</span>" for t in range(0, int(seconds) + 1, 10))
    band = "".join(f'<div data-section="{name}" style="{left(ts[a] - .01)};width:{(ts[b - 1] - ts[a] + .01) / seconds * 100:.3f}%" '
                   f'title="{name}, {clock(ts[a] - .01)} to {clock(ts[b - 1])}">{name}</div>'
                   for a, b, name in _runs([s.section for s in structure]))
    marks, noted = [], []
    for t, s in zip(ts, structure):
        for kind, strength in (("boundary", s.boundary), ("drop", float(s.drop)),
                               ("vocal-entry", float(s.vocal_entry)), ("vocal-exit", float(s.vocal_exit))):
            if strength > 0:
                size = f";height:{strength * 100:.0f}%" if kind == "boundary" else ""
                label = f"boundary of strength {strength:.1f}" if kind == "boundary" else kind.replace("-", " ")
                marks.append(f'<i data-mark="{kind}" style="{left(t)}{size}" title="{label} at {clock(t)}"></i>')
                noted.append([round(t, 2), label])
    presence = {name: _svg("".join(f'<rect x="{ts[a] - .01:.2f}" y="0" width="{ts[b - 1] - ts[a] + .01:.2f}" height="1"/>'
                                   for a, b, present in _runs([bool(f.layers and getattr(f.layers, name).present) for f in features])
                                   if present), seconds) for name in LAYERS}
    energy = [structure[i].energy for i in drawn]
    tension = [structure[i].tension for i in drawn]
    floor = [0.0] * len(xs)
    weights = [structure[i].focus for i in drawn]
    upto = [[sum(w[:k]) for w in weights] for k in range(4)]
    focus = _svg("".join(_area(xs, upto[k], upto[k + 1], f"g-{name}") for k, name in enumerate(GROUPS)), seconds)
    lead = "".join(f'<polyline data-lane="lead" points="{" ".join(f"{xs[i]:.2f},{1 - voices[drawn[i]].position:.3f}" for i in range(a, b))}"/>'
                   for a, b, heard in _runs([voices[i].level > HEARD for i in drawn]) if heard and b - a > 1)
    data = {"step": STEP / 100, "from": round(xs[0], 2) if xs else 0.0,
            "section": "".join(structure[i].section[0] if structure[i].section != "build" else "u" for i in drawn),
            "energy": [round(v * 100) for v in energy], "tension": [round(v * 100) for v in tension],
            "focus": [[round(v * 100) for v in w] for w in weights],
            "angle": [round(tones[i].angle) if tones[i].confidence > 0 else None for i in drawn],
            "chord": [round(tones[i].chord) if tones[i].confidence > 0 else None for i in drawn],
            "lead": [[round(voices[i].position * 100), round(voices[i].hz)] if voices[i].level > HEARD else None for i in drawn],
            "marks": noted}
    source = features[-1].source if features else "bands"
    rows = [("", "axis", f'<div class="ticks">{ticks}</div>'),
            ("section", "sec", band),
            ("marks", "marks", "".join(marks)),
            ("energy", "curve", _svg(_area(xs, floor, energy, "ink"), seconds)),
            ("tension", "curve", _svg(_area(xs, floor, tension, "ink"), seconds)),
            *((name, "bar", presence[name]) for name in LAYERS),
            ("focus", "focus", focus),
            ("chord", "strip", _strip(ts, tones, "chord", seconds)),
            ("colour", "strip", _strip(ts, tones, "angle", seconds)),
            ("lead", "lead", _svg(lead, seconds))]
    names = "".join(f'<span class="r h-{kind}">{name}</span>' for name, kind, _ in rows)
    lanes = "".join(f'<div class="r h-{kind} {kind}">{inner}</div>' for _, kind, inner in rows)
    return (f'<section class="clip" data-seconds="{seconds:.2f}">'
            f"<h2>{escape(title)}</h2>"
            f'<p class="about">{clock(seconds)}, read from {"the stems" if source == "stems" else "the mix"}</p>'
            f'<div class="controls"><audio controls preload="none" src="{escape(audio_href, quote=True)}"></audio>'
            '<span class="zoom"><button type="button" data-zoom="whole" aria-pressed="true">Whole song</button>'
            '<button type="button" data-zoom="minute" aria-pressed="false">One minute</button></span></div>'
            '<p class="readout">Click the timeline to play from there. Point at it to read the values.</p>'
            f'<div class="tracks"><div class="names">{names}</div>'
            f'<div class="lanes"><div class="inner">{lanes}<div class="playhead"></div><div class="cross" hidden></div></div></div></div>'
            f'<script type="application/json">{json.dumps(data, separators=(",", ":"))}</script>'
            "</section>")


STYLE = """
:root{color-scheme:light dark;--surface:#fcfcfb;--raised:#f1f0ec;--ink:#0b0b0b;--ink2:#52514e;--line:#d9d8d2;
--quiet:#b5b4ae;--groove:#2a78d6;--breakdown:#eb6834;--build:#1baf7a;--drop:var(--ink);
--drums:#e87ba4;--harmony:#008300;--lead:#6250d6;--names:64px}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--raised:#262624;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37;
--quiet:#5f5e59;--groove:#3987e5;--breakdown:#d95926;--build:#199e70;
--drums:#d55181;--harmony:#008300;--lead:#9085e9}}
*{box-sizing:border-box}
body{margin:0;padding:24px 16px 64px;background:var(--surface);color:var(--ink);font:14px/1.45 system-ui,sans-serif}
main{max-width:1500px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:17px;margin:0}
p{margin:0}
.intro,.about,.legend{color:var(--ink2)}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin:14px 0 0;padding:0;list-style:none;font-size:12px}
.legend b{font-weight:600;color:var(--ink)}
.legend li{min-width:0}
.legend span{margin-right:8px}
.legend .k{display:inline-flex;align-items:center;gap:5px;white-space:nowrap}
.sw{width:12px;height:8px;border-radius:2px;flex:none}
.clip{margin-top:36px;padding-top:20px;border-top:1px solid var(--line)}
.controls{display:flex;flex-wrap:wrap;align-items:center;gap:10px 16px;margin:10px 0 8px}
audio{flex:1 1 280px;max-width:520px;min-width:0;height:36px}
.zoom{display:inline-flex}
.zoom button{font:inherit;font-size:12px;padding:5px 10px;color:var(--ink2);background:none;border:1px solid var(--line);cursor:pointer}
.zoom button:first-child{border-radius:6px 0 0 6px}
.zoom button:last-child{border-radius:0 6px 6px 0;border-left:0}
.zoom button[aria-pressed=true]{background:var(--raised);color:var(--ink);font-weight:600}
.readout{min-height:3em;font-size:12px;color:var(--ink2);font-variant-numeric:tabular-nums}
.tracks{display:grid;grid-template-columns:var(--names) minmax(0,1fr)}
.names span{display:flex;align-items:center;font-size:11px;color:var(--ink2);white-space:nowrap}
.r{margin-bottom:4px}
.h-axis{height:16px}.h-sec{height:26px}.h-marks{height:24px}.h-curve{height:30px}.h-bar{height:10px}
.h-focus{height:40px}.h-strip{height:10px}.h-lead{height:36px}
.lanes{position:relative;overflow:hidden;cursor:crosshair;touch-action:pan-y}
.inner{position:relative;width:100%}
.inner>.r{position:relative}
svg{display:block;width:100%;height:100%}
.curve,.lead,.focus,.inner>.h-bar{background:var(--raised)}
.ticks span{position:absolute;top:0;font-size:10px;line-height:16px;color:var(--ink2);padding-left:3px;border-left:1px solid var(--line);font-variant-numeric:tabular-nums}
.ticks .minor{display:none}
.zoomed .ticks .minor{display:block}
@media (min-width:1100px){.ticks .minor{display:block}}
.sec div{position:absolute;top:0;bottom:0;border-top:8px solid var(--quiet);padding:1px 3px 0;font-size:11px;line-height:16px;color:var(--ink2);
overflow:hidden;white-space:nowrap;box-shadow:inset 1px 0 var(--surface)}
.sec [data-section=groove]{border-color:var(--groove)}
.sec [data-section=breakdown]{border-color:var(--breakdown)}
.sec [data-section=build]{border-color:var(--build)}
.sec [data-section=drop]{border-color:var(--drop)}
.marks{border-bottom:1px solid var(--line)}
.marks i{position:absolute;bottom:0;width:2px;margin-left:-1px;background:var(--ink2)}
.marks [data-mark=drop]{top:0;width:3px;margin-left:-1.5px;background:var(--ink)}
.marks [data-mark=drop]::before{content:"";position:absolute;top:0;left:-3.5px;border:5px solid transparent;border-top:7px solid var(--ink);border-bottom:0}
.marks [data-mark=vocal-entry],.marks [data-mark=vocal-exit]{top:3px;bottom:auto;width:0;height:0;margin:0;background:none;border:5px solid transparent}
.marks [data-mark=vocal-entry]{border-left:7px solid var(--lead);border-right:0}
.marks [data-mark=vocal-exit]{border-right:7px solid var(--lead);border-left:0;margin-left:-7px}
polygon.ink{fill:var(--ink2)}
.h-bar rect{fill:var(--ink2)}
.g-drums{fill:var(--drums)}.g-harmony{fill:var(--harmony)}.g-lead{fill:var(--lead)}
.focus polygon{stroke:var(--surface);stroke-width:1;vector-effect:non-scaling-stroke}
polyline{fill:none;stroke:var(--ink);stroke-width:1.5;stroke-linejoin:round;vector-effect:non-scaling-stroke}
.playhead,.cross{position:absolute;top:0;bottom:0;width:0;pointer-events:none}
.playhead{border-left:2px solid var(--ink);margin-left:-1px}
.cross{border-left:1px dashed var(--ink2)}
@media (max-width:520px){:root{--names:50px}.names span{font-size:10px}}
"""

LEGEND = (
    '<ul class="legend">'
    "<li><b>Section</b> "
    + "".join(f'<span class="k"><i class="sw" style="background:var(--{name})"></i>{name}</span>' for name in SECTIONS)
    + "</li><li><b>Focus</b> "
    + "".join(f'<span class="k"><i class="sw" style="background:var(--{name})"></i>{name}</span>' for name in GROUPS)
    + "</li><li><b>Marks</b> <span>short line: a boundary, as tall as it is strong</span>"
    "<span>full line with a wedge: a drop</span><span>arrows: a voice comes in, a voice leaves</span></li>"
    "<li><b>Chord, colour</b> <span>the hue is the place on the circle of fifths; faint where the reading is only held</span></li>"
    "</ul>")

SCRIPT = """
const NAMES = {q: 'quiet', g: 'groove', b: 'breakdown', u: 'build', d: 'drop'};
const clock = t => Math.floor(t / 60) + ':' + (t % 60).toFixed(1).padStart(4, '0');
const clips = [...document.querySelectorAll('.clip')];
for (const clip of clips) {
  const audio = clip.querySelector('audio'), lanes = clip.querySelector('.lanes'), inner = clip.querySelector('.inner');
  const head = clip.querySelector('.playhead'), cross = clip.querySelector('.cross'), out = clip.querySelector('.readout');
  const data = JSON.parse(clip.querySelector('script').textContent), seconds = +clip.dataset.seconds;
  let zoom = 1, pointed = null, touched = false;
  const describe = t => {
    const i = Math.min(data.section.length - 1, Math.max(0, Math.round((t - data.from) / data.step)));
    if (i < 0 || !data.section.length) return clock(t);
    const f = data.focus[i], figure = ['drums', 'harmony', 'lead'][f.indexOf(Math.max(...f))];
    const parts = [clock(t), NAMES[data.section[i]], 'energy ' + data.energy[i] + '%', 'tension ' + data.tension[i] + '%',
      'figure: ' + figure + ' (drums ' + f[0] + '%, harmony ' + f[1] + '%, lead ' + f[2] + '%)',
      data.angle[i] === null ? 'colour held' : 'colour ' + data.angle[i] + '\\u00b0, chord ' + data.chord[i] + '\\u00b0',
      data.lead[i] ? 'lead at ' + data.lead[i][0] + '% of its range, ' + data.lead[i][1] + ' Hz' : 'no lead'];
    const near = data.marks.filter(m => Math.abs(m[0] - t) <= .25).map(m => m[1] + ' at ' + clock(m[0]));
    return parts.concat(near).join(' \\u00b7 ');
  };
  const place = () => {
    const t = audio.currentTime;
    head.style.left = (t / seconds * 100) + '%';
    const shift = Math.min(Math.max(t / seconds * zoom - 1 / 3, 0), zoom - 1);
    inner.style.width = (zoom * 100) + '%';
    inner.style.transform = 'translateX(' + (-shift / zoom * 100) + '%)';
    if (touched || pointed !== null) out.textContent = describe(pointed ?? t);
  };
  const loop = () => { place(); if (!audio.paused) requestAnimationFrame(loop); };
  const at = e => { const r = inner.getBoundingClientRect(); return Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)) * seconds; };
  audio.addEventListener('play', () => { touched = true; clips.forEach(c => { const a = c.querySelector('audio'); if (a !== audio) a.pause(); }); loop(); });
  for (const name of ['seeked', 'timeupdate', 'pause']) audio.addEventListener(name, place);
  lanes.addEventListener('pointermove', e => { pointed = at(e); cross.style.left = (pointed / seconds * 100) + '%'; cross.hidden = false; place(); });
  lanes.addEventListener('pointerleave', () => { pointed = null; cross.hidden = true; place(); });
  lanes.addEventListener('click', e => { touched = true; audio.currentTime = at(e); audio.play(); });
  for (const button of clip.querySelectorAll('.zoom button')) button.addEventListener('click', () => {
    zoom = button.dataset.zoom === 'minute' ? Math.max(1, seconds / 60) : 1;
    clip.classList.toggle('zoomed', zoom > 1);
    clip.querySelectorAll('.zoom button').forEach(b => b.setAttribute('aria-pressed', b === button));
    place();
  });
}
"""


def page(fragments: list[str]) -> str:
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Structure, tone and lead</title>'
            f"<style>{STYLE}</style></head><body><main><h1>Structure, tone and lead</h1>"
            '<p class="intro">What the engine reads in each song, every 10 ms, drawn against the song itself. '
            "Play a clip and watch whether the marks fall where you hear them.</p>"
            f'{LEGEND}{"".join(fragments)}</main><script>{SCRIPT}</script></body></html>')


def find(clip: str) -> Path:
    path = Path(clip)
    if path.is_file():
        return path.resolve()
    for candidate in sorted(CORPUS.rglob("*")):
        if candidate.suffix.lower() in AUDIO and clip.lower() in candidate.name.lower():
            return candidate
    raise SystemExit(f"no audio file matching {clip!r} in {CORPUS}")


def defaults() -> list[Path]:
    """The full tracks, then the first clip of every genre."""
    files = sorted(p for p in CORPUS.rglob("*") if p.suffix.lower() in AUDIO)
    full = sorted(p for p in files if p.parent == CORPUS)
    by_genre: dict[str, Path] = {}
    for p in files:
        meta = p.with_suffix(".json")
        if p.parent != CORPUS and meta.exists():
            by_genre.setdefault(json.loads(meta.read_text()).get("genre", "?"), p)
    return full + [by_genre[genre] for genre in sorted(by_genre)]


def one(audio: Path) -> str:
    if cached_stems(audio) is None:
        from structure_report import separate
        separate(audio)
    meta = audio.with_suffix(".json")
    info = json.loads(meta.read_text()) if meta.exists() else {}
    title = info.get("title") or audio.stem.replace("_", " ").capitalize()
    if info.get("genre"):
        title += f" ({info['genre'].replace('_', ' ')})"
    return timeline(analyse(audio), title, quote(os.path.relpath(audio, OUT.parent)))


class Ranged(SimpleHTTPRequestHandler):
    """Static files, with the byte ranges a browser asks for when it seeks in a sound file."""

    def do_GET(self) -> None:
        port = self.server.server_address[1]
        if self.headers.get("Host") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            self.send_error(403)            # a page elsewhere that points its own name here gets nothing
            return
        if self.path == "/":
            self.send_response(302)
            self.send_header("Location", "/" + OUT.relative_to(HERE).as_posix())
            self.end_headers()
            return
        super().do_GET()

    def send_head(self):
        self.left = None
        path = Path(self.translate_path(self.path))
        asked = self.headers.get("Range", "")
        if not path.is_file() or not asked.startswith("bytes="):
            return super().send_head()
        size = path.stat().st_size
        first, _, last = asked[6:].partition("-")
        try:
            start = int(first) if first else max(0, size - int(last))
            end = min(int(last), size - 1) if first and last else size - 1
        except ValueError:
            return super().send_head()
        if not 0 <= start <= end:
            self.send_error(416)
            return None
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        file = path.open("rb")
        file.seek(start)
        self.left = end - start + 1
        return file

    def end_headers(self) -> None:
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def copyfile(self, source, outputfile) -> None:
        if self.left is None:
            return super().copyfile(source, outputfile)
        while self.left > 0:
            chunk = source.read(min(65536, self.left))
            if not chunk:
                break
            outputfile.write(chunk)
            self.left -= len(chunk)

    def log_message(self, *args) -> None:
        pass


def server(directory: Path, port: int) -> ThreadingHTTPServer:
    """A file server for `directory` on this computer only (127.0.0.1); the caller runs serve_forever()."""
    return ThreadingHTTPServer(("127.0.0.1", port), lambda *args: Ranged(*args, directory=str(directory)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="*")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--serve", type=int, metavar="PORT", help="serve the page that is there instead of writing it")
    args = parser.parse_args()
    if args.serve:
        if not OUT.exists():
            raise SystemExit(f"{OUT.relative_to(HERE.parent)} is not there yet: run this without --serve first")
        print(f"http://127.0.0.1:{args.serve}/", flush=True)
        server(HERE, args.serve).serve_forever()
    files = [find(clip) for clip in args.clips] or defaults()
    if not files:
        raise SystemExit(f"no audio in {CORPUS}: see bench/DATA.md")
    with ProcessPoolExecutor(args.workers) as pool:
        fragments = list(pool.map(one, files))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page(fragments))
    print(f"{len(files)} clips; written to {OUT.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
