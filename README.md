# WLED DJ

A macOS framework for music-reactive [WLED](https://kno.wled.ge) lights. WLED DJ listens to whatever your Mac is playing, on any output, works out what the music is doing, and streams frames to your WLED controller in time with the sound.

**It ships no effects.** It gives you the hard parts (audio capture, analysis, sync, output, safety, a control page) and a small interface for writing your own effect, called a *look*. One calibration look, Sync check, is built in so you can see it working.

It never touches your audio: no virtual audio device, no change of output, no added delay. To stay in sync, the lights wait instead.

## What you get

- **Capture of system audio** on any output (speakers, AirPods, USB, AirPlay), through a small Swift menu-bar app that holds the macOS permission.
- **Analysis every 10 ms:** kicks, snares, hats, bass notes, vocal onsets, tempo and beat; four layers (drums, bass, vocals, other) with level and presence; the song's energy, tension and section (quiet, groove, breakdown, build, drop), boundaries and drops; the harmony's colour; the lead's pitch position.
- **Layer separation** with the [StemgenRT](https://github.com/sweetspotsoundsystem/stemgen-rt) model in a helper process, with a fallback to estimates from the mix when the Mac is too busy.
- **Sync:** the lights are delayed by the latency macOS reports for the current output, with a fine trim per device.
- **Output:** DDP to WLED, found by mDNS; WLED's own state is restored on quit.
- **A flash limit** for every look: at most three full-strip flashes a second by default, following WCAG 2.3.1.
- **A studio page** in the browser: live preview, look picker, brightness, blackout, sync trim, flash limit.
- **Tools for look authors:** perceptual colour (OKLab) and a canvas to paint on, offline renders of a look on your own music as video, and numbers read from those renders.

Requires macOS 14.2 or later (for the system-audio tap), Python 3.13 through [uv](https://docs.astral.sh/uv/), Xcode command-line tools to build the app, and `ffmpeg` for the render tools.

## Start

```sh
git clone https://github.com/haslanli/WLED-DJ.git
cd WLED-DJ
uv sync
app/build.sh
open "WLED DJ.app"
```

A beacon icon appears in the menu bar and the studio opens in your browser.

- **Permissions:** macOS asks whether WLED DJ may record system audio and find devices on your local network. Allow both. If you missed a prompt, enable WLED DJ under System Settings → Privacy & Security → Screen & System Audio Recording (or → Local Network), then choose Restart Engine in the menu.
- **Model:** the separation model (32 MB) is downloaded once on first start. Until then, and whenever it cannot be fetched, everything runs without it.
- **Your lights:** controllers are found automatically. If none is, type the address (for example `192.168.1.50`) under Lights in the studio.
- **Check it works:** pick *Sync check* and play something with a clear kick. The whole strip flashes on every kick.

The app is ad-hoc signed, so a rebuilt app makes macOS ask for the permissions again.

## Write a look

A look is one class in one Python file. Put the file in `~/Library/Application Support/WLED DJ/looks/` and restart the engine; it appears in the studio's Look picker.

```python
import numpy as np

from wled_dj.looks.canvas import Canvas, mass, pulse
from wled_dj.looks.colour import RING


class Glow:
    id = "glow"          # saved in the settings: keep it stable
    name = "Glow"        # shown in the studio

    def __init__(self, pixels: int):
        self.canvas = Canvas(pixels)
        self.kick_at = -1e9

    def render(self, f, lead: float) -> np.ndarray:
        """One frame for the Features `f` (one arrives every 10 ms). Returns (pixels, 3) float32 in 0-1."""
        if f.structure is None or f.tone is None or f.silent:
            self.canvas.clear()
            return self.canvas.values()
        if f.kick is not None:
            self.kick_at = f.t
        place = f.tone.angle / 360                  # the song's colour, as a place on the palette ring
        self.canvas.fill(*RING.colour(place, .1 + .3 * f.structure.energy))
        hit = pulse(f.t - self.kick_at, .05, .15)   # 1 on a kick, held 50 ms, then dying away
        self.canvas.paint(mass(self.canvas.r, .2, .1) * hit, *RING.colour(place, .8, chroma=.5))
        return self.canvas.values()
```

This is `examples/looks/glow.py`. The rules:

- `id` and `name` are strings; `__init__(pixels)`; `render(features, lead)` returns a `(pixels, 3)` float32 array in 0–1.
- `render` is called once per 10 ms hop and must be fast (well under a millisecond).
- A file that fails to import is logged and skipped; it never stops the app.
- The flash limit applies to whatever you return. If your look carries a `flash_limit` attribute, the engine keeps it at the studio's setting, so you can budget large gestures yourself (`wled_dj.flash.FlashBudget`).

### What `features` holds

| Field | What |
|---|---|
| `t` | Time of the hop, in seconds |
| `kick`, `snare`, `hats`, `bass`, `vocal`, `stab` | An event (`.strength` 0–1, `.pan`) on the hop it happens, else `None` |
| `beat` | `.bpm`, `.phase` (0–1 through the beat), `.confidence`, `.phase_at(time)` |
| `layers` | `.drums`, `.bass`, `.vocals`, `.other`: each with `.level_db`, `.rel_db` (against the mix), `.present` |
| `structure` | `.energy` and `.tension` (0–1), `.section`, `.boundary` (strength, on the hop), `.drop`, `.vocal_entry`, `.vocal_exit`, `.focus` (weights of drums, harmony, lead) |
| `tone` | `.angle` (the song's colour, degrees on the circle of fifths), `.chord`, `.confidence` |
| `voice` | The lead: `.position` (0–1 in its own range), `.level`, `.vocal` |
| `silent`, `source` | Whether the hop is silence; `"stems"` or `"bands"` (where the layers came from) |

`layers`, `structure`, `tone` and `voice` can be `None` (a silence record): draw dark.

Two things real music will teach you quickly: the detectors fire several times a second and are sometimes wrong, and continuous readings such as the lead's position wobble. Drive a look from smoothed values and envelopes, not one shape per event.

### Tools for building one

```sh
uv run python -m wled_dj --open --looks examples/looks        # the engine with a folder of looks, without the app
uv run python tests/fakes/run_demo.py                         # the studio with fake audio and a fake controller
uv run python bench/render_look.py --look glow --looks examples/looks song.wav    # a look on a file: video, numbers, a still
uv run python bench/structure_view.py song.wav                # a page showing what the analysis reads in a song
uv run python bench/measure_engine.py --look sync-check       # timing of the real engine on fakes
```

`render_look.py` writes to `bench/out/looks/`; the studio's **Compare the looks** page (`/looks`) plays the renders side by side with their numbers: full-strip flashes in the busiest second, the fastest blinking of any one spot, average lightness, how much of the strip is lit. `wled_dj.looks.colour` and `wled_dj.looks.canvas` let a look work in lightness, chroma and hue as the eye sees them and convert once.

## The studio

- **Blackout** (or Space) blanks the lights; press again to bring them back.
- **Live** shows the strips, the beat, a meter per kind of hit and per layer, and the song's section, energy and tension.
- **Look** picks the look. **Brightness** is the master level.
- **Sync → Fine trim** nudges the lights by up to 150 ms, remembered per output device.
- **Analysis → Layer separation** turns the separation model on or off (about one processor core).
- **Safety → Flash limit** is the most full-strip flashes a second the lights may make, whatever the look does. Off removes the limit; fast full-strip flashing can trigger seizures in people with photosensitive epilepsy.

When the music stops for 8 seconds the lights fade and WLED returns to its own effect.

## How the sync works

Sound takes time to reach you: about 20 ms from the MacBook's speakers, about 170 ms through AirPods, around 2 seconds over AirPlay. WLED DJ reads that latency from macOS and delays the lights by it. On wired outputs, where the lights cannot be earlier than the music they react to, processing is kept to a few milliseconds and `lead` tells a look how late it runs, so beat-locked elements can fire early.

## If something is wrong

| You see | Do this |
|---|---|
| "No audio is arriving" | Play some music. Check the Screen & System Audio Recording permission, then Restart Engine. |
| "Can't reach your lights" | Check the controller is powered and on the same network, or connect by address. Check the Local Network permission. |
| "Your lights are showing another live stream" | Something else is streaming to WLED. WLED DJ takes over when it stops. |
| "Layer separation paused: this Mac is busy" | The model could not keep up; WLED DJ follows the mix and retries by itself. |
| Your look is missing | Menu-bar icon → Show Log (`~/Library/Logs/WLED DJ/engine.log`) says why its file was not loaded. |

Settings live in `~/Library/Application Support/WLED DJ/settings.json`.

## Layout

```
WLED DJ.app (menu bar) ── starts ──► engine (python -m wled_dj)
                                      ├─ wled-capture (Core Audio process tap)
                                      ├─ analysis: 10 ms hops → Features
                                      │    └─ stem helper (python -m wled_dj.stems): StemgenRT
                                      ├─ look → compositor (master, blackout, fade, flash limit) → timed sender
                                      ├─ DDP → WLED
                                      └─ studio: http://127.0.0.1:8770 (local only)
```

| Folder | What |
|---|---|
| `app/` | Swift package: the menu-bar app and the capture helper |
| `engine/wled_dj/` | The engine: capture client, analysis, separation (`stems/`), looks, compositor, sync, output, studio server |
| `studio/` | The studio pages (plain HTML, CSS and JS) |
| `examples/looks/` | An example look |
| `bench/` | Offline tools: renders, reports, scoring, timing. See `bench/DATA.md` for where data goes |
| `tests/` | Engine tests, with a fake capture helper and a fake WLED |

```sh
uv run pytest                     # engine tests
swift test --package-path app     # app and capture-helper logic
```

## Credits and licence

WLED DJ is released under the MIT licence (see `LICENSE`).

Layer separation uses the [StemgenRT](https://github.com/sweetspotsoundsystem/stemgen-rt) model weights by Axel Delafosse, licensed CC BY-NC-SA 4.0 (non-commercial, share-alike), trained on MUSDB18-HQ and MoisesDB. WLED DJ downloads them unchanged at first start and does not redistribute them; their licence, not MIT, governs their use.
