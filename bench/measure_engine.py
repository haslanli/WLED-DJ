"""Timing of the real engine on fakes: how long a hop takes from capture to frame, and how often its stems are late.

    uv run python bench/measure_engine.py [--seconds 60] [--block 128] [--latency-ms 22.1]
        [--stems real|fake|off] [--models DIR] [--look ID]

The engine is set up as tests/fakes/run_demo.py sets it up (fake capture, fake WLED, no studio) and asked for its
status once a second. The defaults are the hard case: the 128-sample blocks of a 192 kHz output, 22 ms of output
latency, the real separation model (from bench/data/stemgenrt).

To measure another version of the engine, unpack it and put it first on PYTHONPATH; the first line printed says
which engine ran:

    mkdir /tmp/other && git archive OTHER_BRANCH engine | tar -x -C /tmp/other
    PYTHONPATH=/tmp/branch-point/engine uv run python bench/measure_engine.py"""
from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
FAKES = HERE.parent / "tests" / "fakes"
sys.path.insert(0, str(FAKES.parent))
from fakes.fake_wled import FakeWled  # noqa: E402

import wled_dj  # noqa: E402
from wled_dj.config import ControllerConfig, Settings  # noqa: E402
from wled_dj.service import Engine  # noqa: E402
from wled_dj.stems.link import StemLink  # noqa: E402
from wled_dj.stems.models import verified  # noqa: E402

WARM_UP = 8                         # polls left out of the timing: the model loads and the first stems arrive


def summarise(polls: list[dict], warm_up: int = WARM_UP) -> dict:
    """polls: Engine.status(), once a second. Pauses are counted over all of them, the rest after the warm-up."""
    states = [poll["layers"]["state"] for poll in polls]
    timed = polls[warm_up:]
    late = [poll["layers"]["late_share"] for poll in timed if poll["layers"]["state"] == "on"]
    return {"capture_to_frame_ms": statistics.median(poll["sync"]["processing_ms"] for poll in timed),
            "late_mean": round(statistics.fmean(late), 4) if late else 0.0,
            "late_highest": max(late, default=0.0),
            "send_p99_ms": max(poll["output"]["interval_p99_ms"] for poll in timed),
            "pauses": sum(before != "paused" and after == "paused" for before, after in zip(states, states[1:])),
            "from_stems": sum(poll["layers"]["source"] == "stems" for poll in timed) / len(timed)}


def parser() -> argparse.ArgumentParser:
    options = argparse.ArgumentParser()
    options.add_argument("--seconds", type=int, default=60)
    options.add_argument("--block", default="128")
    options.add_argument("--latency-ms", default="22.1")
    options.add_argument("--stems", choices=("real", "fake", "off"), default="real")
    options.add_argument("--models", type=Path, default=HERE / "data" / "stemgenrt")
    options.add_argument("--look", default="sync-check", help="the look the engine renders while it is measured")
    return options


def settings_for(args: argparse.Namespace, host: str) -> Settings:
    return Settings(controllers=[ControllerConfig(host)], look=args.look)


def main() -> None:
    args = parser().parse_args()
    stems = None
    if args.stems == "fake":
        stems = StemLink([sys.executable, str(FAKES / "fake_stems.py")])
    elif args.stems == "real":
        if not verified(args.models):
            raise SystemExit(f"no verified model files in {args.models}")
        stems = StemLink([sys.executable, "-m", "wled_dj.stems", str(args.models)])
    fake = FakeWled().start()
    state = Path(tempfile.mkdtemp(prefix="wled-dj-measure-"))
    engine = Engine(settings_for(args, fake.host), state / "settings.json",
                    [sys.executable, str(FAKES / "fake_capture.py"), "--latency-ms", args.latency_ms, "--block", args.block],
                    ddp_port=fake.ddp_port, http_port=fake.http_port, discover=lambda timeout=3.0: [], stems=stems)
    print(f"engine: {Path(wled_dj.__file__).parent}; look {args.look}, stems {args.stems}, blocks of {args.block}, "
          f"{args.latency_ms} ms of output latency, {args.seconds} s", flush=True)
    engine.start()
    polls = []
    try:
        for _ in range(args.seconds):
            time.sleep(1.0)
            polls.append(engine.status())
    finally:
        engine.stop()
        fake.stop()
    result = summarise(polls)
    print(f"capture to frame {result['capture_to_frame_ms']:.1f} ms (median); "
          f"stems late {result['late_mean']:.1%} on average, {result['late_highest']:.1%} at most; "
          f"frames p99 {result['send_p99_ms']:.1f} ms apart; paused {result['pauses']} times; "
          f"layers from the stems {result['from_stems']:.0%} of the time")


if __name__ == "__main__":
    main()
