"""The timing summary of bench/measure_engine.py, on made-up status readings."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

from measure_engine import summarise  # noqa: E402


def poll(processing, state="on", late=0.0, source="stems", p99=13.0):
    return {"sync": {"processing_ms": processing}, "output": {"interval_p99_ms": p99},
            "layers": {"state": state, "late_share": late, "source": source}}


def test_the_summary_reads_the_polls():
    polls = ([poll(30.0, "starting", source="bands")] * 2                 # warm-up: left out of the timing
             + [poll(12.0, late=.01), poll(13.0, late=.03), poll(14.0, late=.02, p99=15.5)]
             + [poll(11.0, "paused", late=.6, source="bands")] * 2 + [poll(12.5, late=0.0)])
    result = summarise(polls, warm_up=2)
    assert result["capture_to_frame_ms"] == 12.25                         # the median of the six after the warm-up
    assert result["late_mean"] == 0.015 and result["late_highest"] == .03   # only while separation is on
    assert result["send_p99_ms"] == 15.5 and result["pauses"] == 1
    assert result["from_stems"] == 4 / 6


def test_a_run_without_stems_has_no_late_share():
    result = summarise([poll(9.0, "off", source="bands")] * 4, warm_up=1)
    assert result["late_mean"] == 0.0 and result["late_highest"] == 0.0 and result["pauses"] == 0 and result["from_stems"] == 0.0


def test_the_look_to_measure_can_be_chosen():
    from measure_engine import parser, settings_for
    assert settings_for(parser().parse_args([]), "127.0.0.1").look == "sync-check"
    chosen = settings_for(parser().parse_args(["--look", "glow", "--seconds", "5"]), "127.0.0.1")
    assert chosen.look == "glow" and chosen.controllers[0].host == "127.0.0.1"
