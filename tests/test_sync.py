import numpy as np
import pytest

from wled_dj.output.sender import Sender, TimedFrame
from wled_dj.sync import SyncModel

FRAME = np.zeros((78, 3), np.uint8)


def settled(latency_ms, processing_ms=15.0):
    sync = SyncModel()
    sync.set_output(latency_ms, now=-10.0)
    for _ in range(200):
        sync.observe_processing(processing_ms)
    return sync


def test_offset_airpods():
    sync = settled(170.7)
    assert sync.offset(0) == pytest.approx(.1607, abs=1e-3)
    assert sync.delay(0) == pytest.approx(.1457, abs=1e-3)
    assert sync.lead(0) == 0


def test_offset_wired():
    sync = settled(22.1)
    assert sync.offset(0) == pytest.approx(.0121, abs=1e-3)
    assert sync.lead(0) == pytest.approx(.0029, abs=1e-3)
    assert sync.delay(0) == 0


def test_trim_clamped():
    sync = settled(170.7)
    sync.set_trim(400)
    assert sync.trim_ms == 150
    assert sync.offset(0) == pytest.approx(.3107, abs=1e-3)


def test_ramp():
    sync = settled(22.1)
    sync.set_output(170.7, now=0.0)
    middle = (.0121 + .1607) / 2
    assert abs(sync.offset(.5) - middle) <= .1 * middle
    assert sync.offset(1.0) == pytest.approx(.1607, abs=1e-6)
    assert sync.offset(5.0) == pytest.approx(.1607, abs=1e-6)


class Recorder:
    def __init__(self, result=True):
        self.calls = []
        self.result = result
        self.now = 0.0

    def __call__(self, rgb):
        self.calls.append((self.now, rgb))
        return self.result


def drive(sender, recorder, until, step=.001, start=0.0):
    t = start
    while t <= until + 1e-9:
        recorder.now = t
        sender.pump(t)
        t = round(t + step, 6)


def test_sender_order_and_skips():
    rec = Recorder()
    sender = Sender(rec, fps=77)
    for i in range(200):
        sender.submit(TimedFrame(i * .01, i * .01, np.full((78, 3), i % 256, np.uint8)))
    drive(sender, rec, 2.0)
    sent = [int(rgb[0, 0]) for _, rgb in rec.calls]
    assert sent == sorted(sent) and 150 <= len(sent) <= 156
    assert sender.stats()["skipped"] > 0 and sender.stats()["sent"] == len(sent)


def test_sender_waits_for_due():
    rec = Recorder()
    sender = Sender(rec, fps=77)
    sender.submit(TimedFrame(1.1, 1.0, FRAME))
    drive(sender, rec, 1.099, start=1.0)
    assert rec.calls == []
    drive(sender, rec, 1.1, start=1.1)
    assert len(rec.calls) == 1


def test_failed_transmit_backs_off():
    rec = Recorder(result=False)
    sender = Sender(rec, fps=77)
    for i in range(100):
        sender.submit(TimedFrame(i * .01, i * .01, FRAME))
    drive(sender, rec, 1.0)
    attempts = [t for t, _ in rec.calls]
    assert len(attempts) >= 2 and min(np.diff(attempts)) >= .1 - 1e-9
    assert sender.stats()["errors"] == len(attempts)


def test_inactive_discards():
    rec = Recorder()
    sender = Sender(rec, fps=77)
    sender.active = False
    for i in range(10):
        sender.submit(TimedFrame(i * .01, i * .01, FRAME))
    drive(sender, rec, .2)
    assert rec.calls == []
    sender.active = True
    drive(sender, rec, .4, start=.2)
    assert rec.calls == []


def test_latency_and_interval_stats():
    rec = Recorder()
    sender = Sender(rec, fps=50)
    for i in range(100):
        sender.submit(TimedFrame(i * .01 + .1, i * .01, FRAME))
    drive(sender, rec, 1.2)
    stats = sender.stats()
    assert 100 <= stats["latency_ms"] < 120   # due at +100 ms, sent on the next 20 ms slot
    assert stats["interval_p99_ms"] == pytest.approx(20, abs=1.5)


def test_sender_thread_sends_in_real_time():
    import time
    rec = Recorder()
    sender = Sender(rec, fps=77)
    sender.start()
    try:
        now = time.monotonic()
        for i in range(30):
            sender.submit(TimedFrame(now + i * .01, now + i * .01, FRAME))
        time.sleep(.4)
    finally:
        sender.stop()
    assert 20 <= len(rec.calls) <= 30


def test_sender_survives_a_transmit_exception():
    calls = []

    def transmit(rgb):
        calls.append(rgb)
        if len(calls) <= 2:
            raise RuntimeError("socket exploded")
        return True

    sender = Sender(transmit, fps=77)
    for i in range(100):
        sender.submit(TimedFrame(i * .01, i * .01, FRAME))
    t = 0.0
    while t <= 1.0:
        sender.pump(t)
        t = round(t + .001, 6)
    stats = sender.stats()
    assert stats["errors"] == 2 and stats["sent"] > 0


def test_sender_thread_keeps_running_after_exceptions():
    import time
    state = {"n": 0}

    def transmit(rgb):
        state["n"] += 1
        if state["n"] <= 3:
            raise OSError(64, "Host is down")
        return True

    sender = Sender(transmit, fps=77)
    sender.start()
    try:
        now = time.monotonic()
        for i in range(80):
            sender.submit(TimedFrame(now + i * .01, now + i * .01, FRAME))
        time.sleep(.8)
        assert sender._thread.is_alive() and sender.stats()["sent"] > 0
    finally:
        sender.stop()


def test_sender_never_sends_older_audio_after_newer():
    # A trim pulled earlier makes new frames due before older queued ones: the old ones must be dropped.
    rec = Recorder()
    sender = Sender(rec, fps=77)
    for i in range(20):                                   # audio 0.00..0.19, due +0.20
        sender.submit(TimedFrame(.2 + i * .01, i * .01, np.full((78, 3), i, np.uint8)))
    for i in range(20, 60):                               # audio 0.20..0.59, due only +0.05 now
        sender.submit(TimedFrame(.05 + i * .01, i * .01, np.full((78, 3), i, np.uint8)))
    drive(sender, rec, 1.0)
    sent = [int(rgb[0, 0]) for _, rgb in rec.calls]
    assert sent == sorted(sent) and len(set(sent)) == len(sent)
