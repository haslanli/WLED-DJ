import os
from pathlib import Path
import sys
import time

import numpy as np
import pytest

from wled_dj.analysis.resample import Resampler
from wled_dj.stems import link as stem_link
from wled_dj.stems import protocol
from wled_dj.stems.link import StemLink

FAKE = str(Path(__file__).parent / "fakes" / "fake_stems.py")
RNG = np.random.default_rng(7)


def wait_for(predicate, timeout, step=.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return False


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.fixture
def links():
    made = []

    def make(*options, clock=time.monotonic, wait=True):
        link = StemLink([sys.executable, FAKE, *options], clock=clock)
        made.append(link)
        link.start()
        if wait:
            assert wait_for(lambda: link.state == "on", 5)
        return link

    yield make
    for link in made:
        link.stop()


def audio(hops):
    return (RNG.standard_normal((480 * hops, 2)) * .1).astype(np.float32)


def test_protocol_round_trip():
    chunk = RNG.standard_normal((2, 128)).astype(np.float32)
    stems = RNG.standard_normal((3, 128)).astype(np.float32)
    frame = protocol.pack_in(41, chunk)
    assert len(frame) == protocol.IN_BYTES and protocol.unpack_in(frame)[0] == 41
    assert np.array_equal(protocol.unpack_in(frame)[1], chunk)
    reply = protocol.pack_out(protocol.READY, stems)
    assert len(reply) == protocol.OUT_BYTES and protocol.unpack_out(reply)[0] == -1
    assert np.array_equal(protocol.unpack_out(reply)[1], stems)


def test_stem_hops_are_aligned(links):
    link = links()
    x = audio(20)
    mono = Resampler().push(x).mean(axis=1)
    reference = np.concatenate((np.zeros(256, np.float32), mono))     # a stem hop trails the mix by 256 samples
    hops = []
    for n in range(20):
        assert link.push(x[n * 480:(n + 1) * 480], 100.0 + n * .01) == n
        hops += link.collect(.2)
    assert [hop.index for hop in hops] == list(range(20))
    for n, hop in enumerate(hops):
        wanted = reference[441 * n:441 * (n + 1)]
        assert hop.drums.shape == (441,) and hop.drums.dtype == np.float32
        assert np.allclose(hop.drums, wanted, atol=1e-5) and np.allclose(hop.mix, wanted, atol=1e-5)
        assert np.allclose(hop.bass, .5 * wanted, atol=1e-5) and np.allclose(hop.vocals, .25 * wanted, atol=1e-5)
        assert np.allclose(hop.other, hop.mix - hop.drums - hop.bass - hop.vocals, atol=1e-6)
        assert abs(hop.t - (100.0 + n * .01 - link.latency_s)) < 1e-9
    assert abs(link.latency_s - (256 / 44100 + Resampler().delay_s)) < 1e-12
    assert link.late_share == 0


def test_on_time_and_late(links):
    link = links("--delay-ms", "30")
    x = audio(3)
    for n in range(3):
        link.push(x[n * 480:(n + 1) * 480], n * .01)
        assert link.collect(.005) == []
    late = link.collect(1.0)
    assert [hop.index for hop in late] == [0, 1, 2]
    assert link.late_share == pytest.approx(3 / 1000)      # late hops as a share of the last ten seconds


def test_persistent_lateness_pauses(links):
    link = links("--delay-ms", "30")
    pid = link.pid
    x = audio(1)
    for n in range(120):
        link.push(x, n * .01)
        link.collect(.005)
    assert link.state == "paused" and link.push(x, 2.0) is None and link.collect(.01) == []
    assert wait_for(lambda: not alive(pid), 3)


def late_hops(link, count):
    """Push `count` hops in real time, each with a wait its answer cannot make."""
    x = audio(1)
    for n in range(count):
        time.sleep(.01)
        link.push(x, n * .01)
        link.collect(.001)


def test_a_burst_of_late_hops_does_not_pause(links):
    link = links("--delay-ms", "2")              # keeps up with real time, but every answer misses a 1 ms wait
    late_hops(link, 80)                          # 0.8 s of late answers in a row, as after one stall
    assert link.state == "on" and link.late_share == pytest.approx(.08)


def test_a_helper_that_is_late_more_often_than_not_pauses(links, monkeypatch):
    monkeypatch.setattr(stem_link, "LATE_WINDOW", 100)       # judge over one second, to keep the test short
    link = links("--delay-ms", "2")
    late_hops(link, 40)
    assert link.state == "on" and link.late_share == pytest.approx(.4)
    late_hops(link, 40)
    assert link.state == "paused"


def test_one_short_stall_does_not_pause(links):
    link = links("--stall-after", "100", "--stall-ms", "150")
    x = audio(1)
    for n in range(300):
        link.push(x, n * .01)
        link.collect(.01)
    assert link.state == "on" and 0 < link.late_share < .05


def test_frozen_helper_never_blocks_push(links):
    link = links("--freeze-after", "5")
    x = audio(1)
    slowest = 0.0
    for n in range(300):
        started = time.monotonic()
        link.push(x, n * .01)
        slowest = max(slowest, time.monotonic() - started)
        link.collect(0)
    assert slowest < .02
    assert link.state == "paused"


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_dead_helper_restarts_after_the_pause(links):
    clock = Clock()
    link = links("--die-after", "30", clock=clock)
    x = audio(1)
    for n in range(30):
        link.push(x, n * .01)
        link.collect(.02)
    assert wait_for(lambda: link.state == "paused", 2)
    clock.now += 9
    assert link.push(x, 1.0) is None and link.state == "paused"
    clock.now += 2
    assert link.push(x, 1.0) is None and link.state == "starting"
    assert wait_for(lambda: link.state == "on", 5)
    assert link.push(x, 2.0) == 0
    assert [hop.index for hop in link.collect(.2)] == [0]


def test_retry_pauses_grow_to_a_minute(links):
    clock = Clock()
    link = links("--die-after", "3", clock=clock)
    x = audio(1)
    for pause in (10, 20, 40, 60, 60):
        assert wait_for(lambda: link.state == "on", 5)
        link.push(x, 0.0)                                    # four frames: the helper dies after three
        assert wait_for(lambda: link.state == "paused", 2)
        clock.now += pause - 1
        assert link.push(x, 0.0) is None and link.state == "paused"
        clock.now += 2
        assert link.push(x, 0.0) is None and link.state == "starting"


@pytest.mark.parametrize("options, wanted", [(("--ready-after-s", "1"), "starting"), ((), "on"), (("--die-after", "3"), "paused")])
def test_stop_leaves_no_process(links, options, wanted):
    clock = Clock()
    link = links(*options, clock=clock, wait=False)
    x = audio(1)
    if wanted != "starting":
        assert wait_for(lambda: link.state == "on", 5)
        for n in range(5):
            link.push(x, n * .01)
            link.collect(.02)
    assert wait_for(lambda: link.state == wanted, 2)
    pid = link.pid
    link.stop()
    assert link.state == "off" and link.pid is None
    assert pid is None or wait_for(lambda: not alive(pid), 3)
    clock.now += 1000
    assert link.push(x, 9.0) is None and link.state == "off"      # a stopped link never restarts by itself
