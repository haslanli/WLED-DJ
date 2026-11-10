"""Score the engine's own onset detectors on MUSDB18-7 against references from the true stems.

References are non-causal flux peaks on the isolated stems; detections are the engine's causal FrontEnd + OnsetDetector on the mix at 48 kHz.
With --tune, k is chosen per class on the train clips by F0.5 (lights prefer precision),
among values whose train F1 is within 0.05 of the best.

Usage: uv run python bench/score_detection.py [--tune] [--workers 8]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as windows
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from wled_dj.analysis.frontend import HOP, NFFT, RATE, FrontEnd, band
from wled_dj.analysis.onsets import SPECS, OnsetDetector

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "musdb7"
OUT = HERE / "out" / "detection.json"
FT = HOP / RATE
WARMUP, TOL = 1.0, .05
KGRID = [float(k) for k in np.round(np.arange(.5, 8.01, .25), 2)]
# class -> (stem, reference band) as in the benchmark
REFERENCE = {"kick": ("drums", (30, 130)), "snare": ("drums", (1000, 5000)), "hats": ("drums", (7000, 16000)),
             "bass": ("bass", (30, 300)), "vocal": ("vocals", (200, 4000))}
BENCHMARK = {"kick": (.62, .51), "snare": (.71, .66), "hats": (.81, .79), "bass": (.64, .53), "vocal": (.49, .41)}
STREAMS = {"mix": 0, "drums": 1, "bass": 2, "other": 3, "vocals": 4}


def decode(path, stream):
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-map", f"0:a:{stream}", "-f", "f32le", "-ac", "2", "-ar", str(RATE), "-"]
    return np.frombuffer(subprocess.check_output(cmd), "<f4").reshape(-1, 2).copy()


def centred_mag(x):
    x = np.pad(x.mean(axis=1), (NFFT // 2, NFFT // 2))
    n = (len(x) - NFFT) // HOP + 1
    return np.abs(np.fft.rfft(windows(x, NFFT)[::HOP][:n] * np.hanning(NFFT), axis=1))


def reference_events(stem, lo, hi):
    mag = centred_mag(stem)[:, band(lo, hi)]
    lm = np.log1p(1000 * mag)
    nov = np.maximum(np.diff(lm, axis=0, prepend=lm[:1]), 0).sum(axis=1)
    lvl = 10 * np.log10((mag ** 2).sum(axis=1) + 1e-12)
    ok = (nov == maximum_filter1d(nov, 7)) & (nov >= uniform_filter1d(nov, 21) + .1 * nov.max()) \
        & (lvl > lvl.max() - 35) & (nov > 0)
    out = []
    for t in np.flatnonzero(ok) * FT:
        if not out or t - out[-1] >= .05:
            out.append(t)
    return np.array(out)


def match(ref, det):
    ref, det = ref[ref >= WARMUP], np.asarray([t for t in det if t >= WARMUP])
    used = np.zeros(len(det), bool)
    tp = 0
    for r in ref:
        if not len(det):
            break
        d = np.abs(det - r)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= TOL:
            used[j] = True
            tp += 1
    return tp, len(det) - tp, len(ref) - tp


def clip_counts(args):
    """For one clip: {class: {k: (tp, fp, fn)}}."""
    path, ks = args
    stems = {name: decode(path, index) for name, index in STREAMS.items()}
    front = FrontEnd()
    mix = stems["mix"]
    spectra = [front.push(mix[i * HOP:(i + 1) * HOP]) for i in range(len(mix) // HOP)]
    result = {}
    for spec in SPECS:
        stem, (lo, hi) = REFERENCE[spec.name]
        ref = reference_events(stems[stem], lo, hi)
        result[spec.name] = {}
        for k in (ks[spec.name] if isinstance(ks, dict) else ks):
            detector = OnsetDetector(replace(spec, k=k))
            times = [(i + 1) * FT for i, s in enumerate(spectra) if detector.step(s) is not None]
            result[spec.name][k] = match(ref, times)
    return result


def tally(results, ks_by_class):
    totals = {}
    for name in ks_by_class:
        for k in ks_by_class[name]:
            tp = fp = fn = 0
            for r in results:
                a, b, c = r[name][k]
                tp, fp, fn = tp + a, fp + b, fn + c
            totals[(name, k)] = (tp, fp, fn)
    return totals


def fbeta(tp, fp, fn, beta):
    b2 = beta * beta
    return (1 + b2) * tp / max(1, (1 + b2) * tp + b2 * fn + fp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    test = sorted((DATA / "test").glob("*.stem.mp4"))
    train = sorted((DATA / "train").glob("*.stem.mp4"))
    if not test:
        raise SystemExit("run bench/fetch_musdb.py first")
    chosen = {spec.name: spec.k for spec in SPECS}
    with ProcessPoolExecutor(args.workers) as pool:
        if args.tune:
            results = list(pool.map(clip_counts, [(p, KGRID) for p in train]))
            totals = tally(results, {spec.name: KGRID for spec in SPECS})
            chosen = {}
            for spec in SPECS:
                # Precision-leaning (F0.5), but never more than 0.05 F1 below the best F1 on train.
                best_f1 = max(fbeta(*totals[(spec.name, k)], 1) for k in KGRID)
                allowed = [k for k in KGRID if fbeta(*totals[(spec.name, k)], 1) >= best_f1 - .05]
                chosen[spec.name] = max(allowed, key=lambda k: fbeta(*totals[(spec.name, k)], .5))
        results = list(pool.map(clip_counts, [(p, {n: [k] for n, k in chosen.items()}) for p in test]))
    totals = tally(results, {n: [k] for n, k in chosen.items()})
    report = {}
    print(f"{'class':6} {'k':>5} {'F1':>5} {'P':>5} {'R':>5}   benchmark F1/P   pass")
    for spec in SPECS:
        tp, fp, fn = totals[(spec.name, chosen[spec.name])]
        f1, p, r = fbeta(tp, fp, fn, 1), tp / max(1, tp + fp), tp / max(1, tp + fn)
        bf1, bp = BENCHMARK[spec.name]
        ok = p >= bp and f1 >= bf1 - .05
        report[spec.name] = dict(k=chosen[spec.name], f1=round(f1, 3), precision=round(p, 3), recall=round(r, 3),
                                 benchmark_f1=bf1, benchmark_precision=bp, passed=ok)
        print(f"{spec.name:6} {chosen[spec.name]:5.2f} {f1:5.2f} {p:5.2f} {r:5.2f}   {bf1:.2f}/{bp:.2f}        {'yes' if ok else 'NO'}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
