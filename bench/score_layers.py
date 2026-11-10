"""Score the engine's own layer code on full songs with true stems (MUSDB18 test split).

Events: the stem detectors (wled_dj.analysis.onsets.STEM_SPECS) on StemgenRT's streamed stems, against
references from the true stems (10 ms log-flux peaks on a 5 ms grid, matched within 50 ms). Presence: LayerTracker on the stems and on the mix, against the true
stem judged by the same rule the engine uses: its 400 ms level within 15 dB of the mix and above -50 dBFS,
entering after 150 ms of that and leaving after a second without it. (Judged frame by frame instead, every
short gap in a layer counts as an exit the engine is designed not to report.)

With --tune, detector k and the bands' presence thresholds are chosen on the even-numbered songs.
Everything is always reported on the odd-numbered songs.

Usage: uv run python bench/score_layers.py [--tune] [--workers 8]
Needs bench/data/musdb18/test (see bench/DATA.md) and the model files (WLED_DJ_MODELS or the app's folder).
Separated stems are cached in bench/data/stems-cache/musdb18-test/."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as windows
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import resample_poly

from wled_dj.analysis.frontend import HOP, HOP_S, NFFT, FrontEnd
from wled_dj.analysis.layers import (ENTER_S, LAYERS, LEAVE_MARGIN_DB, LEAVE_S, PRESENT_ABS_DB, PRESENT_REL_DB,
                                     LayerTracker, bands_power)
from wled_dj.analysis.onsets import (ABSOLUTE_GATE_DBFS, HISTORY, MIN_HISTORY, PEAK_RELEASE_DB, RELATIVE_GATE_DB,
                                     STEM_SPECS, OnsetDetector)
from wled_dj.stems.link import STEM_HOP
from wled_dj.stems.models import models_dir, verified
from wled_dj.stems.protocol import HOP as MODEL_HOP, LATENCY, RATE

HERE = Path(__file__).resolve().parent
SONGS = HERE / "data" / "musdb18" / "test"
CACHE = HERE / "data" / "stems-cache" / "musdb18-test"
OUT = HERE / "out" / "layers.json"
STREAMS = {"mix": 0, "drums": 1, "bass": 2, "other": 3, "vocals": 4}
REFERENCE = {"kick": (30, 130), "snare": (1000, 5000), "hats": (7000, 16000), "bass": (30, 300), "vocal": (200, 4000),
             "stab": (200, 6000)}
KGRID = [float(k) for k in np.arange(.5, 16.01, .5)]
THRESHOLDS = [float(t) for t in np.arange(-30, -4.9, .5)]
TOLERANCE_S, WARMUP_S, REF_HOP = .05, 1.0, 220
EVENT_LINES = {"kick": (.61, .63), "snare": (.78, .75), "hats": (.79, .83), "bass": (.57, .47)}     # F1, precision
PRESENCE_LINES = {"drums": .88, "bass": .80, "vocals": .88}


def decode(path: Path, stream: int) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-map", f"0:a:{stream}", "-f", "f32le", "-ac", "2", "-ar", str(RATE), "-"]
    return np.frombuffer(subprocess.check_output(cmd), "<f4").reshape(-1, 2).copy()


def separated(path: Path, mix: np.ndarray) -> np.ndarray:
    """(n, 3) float32 mono stems aligned with the mix: from the cache, or by running the model now."""
    target = CACHE / (path.name.replace(".stem.mp4", "") + ".npz")
    if target.exists():
        return np.load(target)["est"].astype(np.float32) / 32767
    from wled_dj.stems.runner import StemModel
    model = StemModel(models_dir())
    count = len(mix) // MODEL_HOP
    est = np.zeros((count * MODEL_HOP, 3), np.float32)
    for k in range(count + 1):                      # the answer to chunk k is the stems of chunk k - 1
        chunk = mix[k * MODEL_HOP:(k + 1) * MODEL_HOP].T if k < count else np.zeros((2, MODEL_HOP), np.float32)
        out = model.step(chunk)
        if k:
            est[(k - 1) * MODEL_HOP:k * MODEL_HOP] = out.T
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, est=np.clip(est * 32767, -32768, 32767).astype(np.int16))
    return est


def reference_events(stem: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Onset times in one band of a true stem (mono): 10 ms log-flux peaks on a 5 ms grid."""
    frames = windows(np.pad(stem, (NFFT // 2, NFFT // 2)), NFFT)[::REF_HOP]
    freq = np.fft.rfftfreq(NFFT, 1 / RATE)
    mask = (freq >= lo) & (freq < hi)
    window = np.hanning(NFFT)
    mag = np.concatenate([np.abs(np.fft.rfft(frames[i:i + 4096] * window, axis=1))[:, mask]
                          for i in range(0, len(frames), 4096)])
    lm = np.log1p(1000 * mag)
    nov = np.zeros(len(lm))
    nov[2:] = np.maximum(lm[2:] - lm[:-2], 0).sum(axis=1)
    level = 10 * np.log10((mag ** 2).sum(axis=1) + 1e-12)
    ok = (nov == maximum_filter1d(nov, 13)) & (nov >= uniform_filter1d(nov, 41) + .1 * maximum_filter1d(nov, 2001)) \
        & (level > level.max() - 40) & (nov > 0)
    out: list[float] = []
    for t in (np.flatnonzero(ok) - 1) * REF_HOP / RATE:
        if not out or t - out[-1] >= .05:
            out.append(float(t))
    return np.array(out)


def rolling(nov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median and MAD of the previous (up to HISTORY) novelty values, as OnsetDetector keeps them."""
    n = len(nov)
    med, mad = np.zeros(n), np.zeros(n)
    for i in range(MIN_HISTORY, min(HISTORY, n)):
        med[i] = np.median(nov[:i])
        mad[i] = np.median(np.abs(nov[:i] - med[i]))
    if n > HISTORY:
        w = windows(nov, HISTORY)[:n - HISTORY]
        m = np.median(w, axis=1)
        med[HISTORY:] = m
        mad[HISTORY:] = np.median(np.abs(w - m[:, None]), axis=1)
    return med, mad


def detect(nov, level, med, mad, floor, k, refractory_s) -> np.ndarray:
    """Hop indices where OnsetDetector would fire (the same rules, vectorised so k can be swept)."""
    peak, p = np.empty(len(nov)), -np.inf
    for i, value in enumerate(level):
        p = max(value, p - PEAK_RELEASE_DB)
        peak[i] = p
    above = (nov > med + np.maximum(k * mad, floor)) & (level > ABSOLUTE_GATE_DBFS) & (level > peak - RELATIVE_GATE_DB)
    above[:MIN_HISTORY] = False
    out, last = [], -10 ** 9
    for i in np.flatnonzero(above[1:] & ~above[:-1]) + 1:
        if (i - last) * HOP_S >= refractory_s:
            out.append(i)
            last = i
    return np.array(out, int)


def match(ref: np.ndarray, det: np.ndarray) -> list[int]:
    ref, det = ref[ref >= WARMUP_S], det[det >= WARMUP_S]
    used = np.zeros(len(det), bool)
    tp = 0
    for r in ref:
        if not len(det):
            break
        d = np.abs(det - r)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= TOLERANCE_S:
            used[j] = True
            tp += 1
    return [tp, len(det) - tp, len(ref) - tp]


def mean_db(x: np.ndarray, hops: int = 40) -> np.ndarray:
    """dB (re a full-scale sine) of the mean power over the last 400 ms, per 10 ms hop of a mono signal."""
    n = len(x) // STEM_HOP
    power = (x[:n * STEM_HOP].reshape(n, STEM_HOP).astype(np.float64) ** 2).mean(axis=1)
    mean = np.convolve(power, np.ones(hops) / hops)[:n]
    return 10 * np.log10(mean / .5 + 1e-12)


def held(rel: np.ndarray, level: np.ndarray, threshold: float, margin: float = LEAVE_MARGIN_DB) -> np.ndarray:
    """Presence per hop by the engine's rule (layers._Presence), from a layer's level relative to the mix."""
    above = (rel > threshold) & (level > PRESENT_ABS_DB)
    below = (rel < threshold - margin) | (level < PRESENT_ABS_DB - margin)
    enter, leave = round(ENTER_S / HOP_S), round(LEAVE_S / HOP_S)
    out = np.zeros(len(rel), bool)
    present, a, b = False, 0, 0
    for i in range(len(rel)):
        a = a + 1 if above[i] else 0
        b = b + 1 if below[i] else 0
        if (not present and a >= enter) or (present and b >= leave):
            present = not present
        out[i] = present
    return out


def song(args) -> dict:
    path, check = args
    true = {name: decode(path, index) for name, index in STREAMS.items()}
    est = separated(path, true["mix"])
    n = min(len(est), len(true["mix"])) // STEM_HOP * STEM_HOP
    mix = true["mix"][:n].mean(axis=1)
    stems = {"drums": est[:n, 0], "bass": est[:n, 1], "vocals": est[:n, 2]}
    stems["other"] = mix - est[:n].sum(axis=1)
    hops = n // STEM_HOP
    result = {"song": path.name.replace(".stem.mp4", ""), "events": {}, "presence": {}}

    # ---- events: every stem detector's novelty and level per hop, then k swept ----
    fronts = {name: FrontEnd(RATE, STEM_HOP, 1) for name in LAYERS}
    detectors = [(stem, OnsetDetector(spec, RATE)) for stem, spec in STEM_SPECS]
    real = [(stem, OnsetDetector(spec, RATE)) for stem, spec in STEM_SPECS] if check else []
    nov = {d.spec.name: np.zeros(hops) for _, d in detectors}
    level = {d.spec.name: np.zeros(hops) for _, d in detectors}
    fired = {d.spec.name: [] for _, d in real}
    tracker = LayerTracker("stems")
    series = {"stems": {name: np.zeros((hops, 2)) for name in LAYERS}, "bands": {name: np.zeros((hops, 2)) for name in LAYERS}}
    present = {"stems": {name: np.zeros(hops, bool) for name in LAYERS}, "bands": {name: np.zeros(hops, bool) for name in LAYERS}}
    for i in range(hops):
        part = {name: stems[name][i * STEM_HOP:(i + 1) * STEM_HOP] for name in LAYERS}
        spectra = {name: front.push(part[name]) for name, front in fronts.items()}
        for stem, detector in detectors:
            nov[detector.spec.name][i], level[detector.spec.name][i], _, _ = detector._novelty_and_level(spectra[stem])
        for stem, detector in real:
            if detector.step(spectra[stem]) is not None:
                fired[detector.spec.name].append(i)
        layers = tracker.step({name: float(np.mean(part[name].astype(np.float64) ** 2)) for name in LAYERS},
                              float(np.mean(mix[i * STEM_HOP:(i + 1) * STEM_HOP].astype(np.float64) ** 2)),
                              spectra["drums"].low_power)
        for name in LAYERS:
            series["stems"][name][i] = layers[name].rel_db, layers[name].level_db
            present["stems"][name][i] = layers[name].present
    delay = LATENCY / RATE
    for stem, detector in detectors:
        name = detector.spec.name
        med, mad = rolling(nov[name])
        if check:
            mine = detect(nov[name], level[name], med, mad, detector.floor, detector.spec.k, detector.spec.refractory_s)
            assert list(mine) == fired[name], f"vectorised detector disagrees with OnsetDetector for {name}"
        ref = reference_events(true[stem][:n].mean(axis=1), *REFERENCE[name])
        result["events"][name] = {str(k): match(ref, (detect(nov[name], level[name], med, mad, detector.floor, k,
                                                             detector.spec.refractory_s) + 1) * HOP_S + delay) for k in KGRID}
        result["events"][name]["current"] = result["events"][name][str(float(detector.spec.k))] \
            if float(detector.spec.k) in KGRID else match(ref, (detect(nov[name], level[name], med, mad, detector.floor,
                                                                        detector.spec.k, detector.spec.refractory_s) + 1) * HOP_S + delay)

    # ---- presence from the mix alone (the engine's 48 kHz path) ----
    mix48 = resample_poly(true["mix"][:n], 160, 147, axis=0).astype(np.float32)
    front, bands = FrontEnd(), LayerTracker("bands")
    for i in range(min(hops, len(mix48) // HOP)):
        layers = bands.step(*bands_power(front.push(mix48[i * HOP:(i + 1) * HOP])))
        for name in LAYERS:
            series["bands"][name][i] = layers[name].rel_db, layers[name].level_db
            present["bands"][name][i] = layers[name].present

    mix_db = mean_db(mix)
    audible = mix_db > -50
    for name in LAYERS:
        true_db = mean_db(true[name][:n].mean(axis=1))
        truth = held(true_db - mix_db, true_db, -15.0, margin=0.0)[audible]
        frame_truth = ((true_db - mix_db > -15) & (true_db > PRESENT_ABS_DB))[audible]
        row = {"truth": [int(truth.sum()), int((~truth).sum())],
               "frame_truth": [int(frame_truth.sum()), int((~frame_truth).sum())]}
        for source in ("stems", "bands"):
            said = present[source][name][audible]
            rel, lvl = series[source][name].T
            if check:
                assert np.array_equal(held(rel, lvl, PRESENT_REL_DB[source][name]), present[source][name]), \
                    f"held() disagrees with LayerTracker for {source} {name}"
            row[source] = [int((said & truth).sum()), int((said & ~truth).sum())]             # hits, false alarms
            frame = ((rel > PRESENT_REL_DB[source][name]) & (lvl > PRESENT_ABS_DB))[audible]
            row[source + "_frame"] = [int((frame & frame_truth).sum()), int((frame & ~frame_truth).sum())]
            row[source + "_sweep"] = []
            for t in THRESHOLDS:
                swept = held(rel, lvl, t)[audible]
                row[source + "_sweep"].append([int((swept & truth).sum()), int((swept & ~truth).sum())])
        result["presence"][name] = row
    return result


def fbeta(tp, fp, fn, beta=1.0):
    b2 = beta * beta
    return (1 + b2) * tp / max(1, (1 + b2) * tp + b2 * fn + fp)


def balanced(hits, false_alarms, present, absent):
    return (hits / max(1, present) + 1 - false_alarms / max(1, absent)) / 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    songs = sorted(SONGS.glob("*.stem.mp4"))
    if not songs:
        raise SystemExit(f"no songs in {SONGS}: see bench/DATA.md")
    missing = [s for s in songs if not (CACHE / (s.name.replace(".stem.mp4", "") + ".npz")).exists()]
    if missing and not verified(models_dir()):
        raise SystemExit(f"{len(missing)} songs are not separated yet and there are no model files in {models_dir()}")
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(song, [(s, i == 0) for i, s in enumerate(songs)]))
    tune, test = results[0::2], results[1::2]
    report = {"songs": len(results), "scored_on": len(test), "events": {}, "presence": {}, "tuned": args.tune}
    ok = True

    print(f"{len(results)} songs; scored on the {len(test)} odd-numbered ones" + (" after tuning on the others" if args.tune else ""))
    print(f"\n{'events':7} {'k':>5} {'F1':>5} {'P':>5} {'R':>5}   line F1/P   pass")
    for _, spec in STEM_SPECS:
        name, key = spec.name, "current"
        if args.tune:
            totals = {k: np.sum([r["events"][name][str(k)] for r in tune], axis=0) for k in KGRID}
            best = max(fbeta(*totals[k]) for k in KGRID)
            chosen = max((k for k in KGRID if fbeta(*totals[k]) >= best - .05), key=lambda k: fbeta(*totals[k], .5))
            key = str(chosen)
        tp, fp, fn = np.sum([r["events"][name][key] for r in test], axis=0)
        f1, precision, recall = fbeta(tp, fp, fn), tp / max(1, tp + fp), tp / max(1, tp + fn)
        line = EVENT_LINES.get(name)
        passed = line is None or (f1 >= line[0] and precision >= line[1])
        ok &= passed
        k = float(key) if args.tune else spec.k
        report["events"][name] = dict(k=k, f1=round(f1, 3), precision=round(precision, 3), recall=round(recall, 3), passed=bool(passed))
        print(f"{name:7} {k:5.2f} {f1:5.2f} {precision:5.2f} {recall:5.2f}   " +
              (f"{line[0]:.2f}/{line[1]:.2f}  " if line else "    -      ") + ("yes" if passed else "NO"))

    print(f"\n{'present':7} {'source':6} {'thr':>6} {'bal.acc':>7} {'hit':>5} {'false':>5}   line   pass   (frame by frame)   suggested thr")
    for name in LAYERS:
        for source in ("stems", "bands"):
            def total(rows, key):
                return np.sum([r["presence"][name][key] for r in rows], axis=0)
            threshold = PRESENT_REL_DB[source][name]
            suggested = None
            if args.tune:
                present_n, absent_n = total(tune, "truth")
                sweep = total(tune, source + "_sweep")
                suggested = THRESHOLDS[int(np.argmax([balanced(h, f, present_n, absent_n) for h, f in sweep]))]
            present_n, absent_n = total(test, "truth")
            hits, alarms = total(test, source)
            score = balanced(hits, alarms, present_n, absent_n)
            frame_score = balanced(*total(test, source + "_frame"), *total(test, "frame_truth"))
            line = PRESENCE_LINES.get(name) if source == "stems" else None
            passed = line is None or score >= line
            ok &= passed
            report["presence"].setdefault(name, {})[source] = dict(
                threshold_db=threshold, balanced_accuracy=round(float(score), 3), hit_rate=round(float(hits / max(1, present_n)), 3),
                false_presence=round(float(alarms / max(1, absent_n)), 3), frame_by_frame=round(float(frame_score), 3),
                passed=bool(passed))
            print(f"{name:7} {source:6} {threshold:6.1f} {score:7.2f} {hits / max(1, present_n):5.2f} {alarms / max(1, absent_n):5.2f}   " +
                  (f"{line:.2f}   " if line else "  -    ") + f"{'yes' if passed else 'NO':6} {frame_score:10.2f}" +
                  (f"{suggested:16.1f}" if suggested is not None else ""))
    if args.tune:
        print("\n--tune: event rows use k chosen on the even-numbered songs. Presence rows use the code's current "
              "thresholds; the last column is what the even-numbered songs suggest.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print("\nall rows pass" if ok else "\nSOME ROWS FAIL")


if __name__ == "__main__":
    main()
