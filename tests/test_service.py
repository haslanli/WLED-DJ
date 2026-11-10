import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from fakes.fake_wled import FakeWled
from wled_dj.capture import AudioBlock
from wled_dj.config import ControllerConfig, Settings, app_support_dir, logs_dir
from wled_dj.service import Engine

FAKE = str(Path(__file__).parent / "fakes" / "fake_capture.py")
ROOT = Path(__file__).resolve().parents[1]


def wait_for(predicate, timeout, step=.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return False


@pytest.fixture
def fake():
    f = FakeWled().start()
    yield f
    f.stop()


def make_engine(tmp_path, fake, scenario="music", *extra, handback=8.0, start=True):
    settings = Settings(controllers=[ControllerConfig(fake.host)], handback_after_s=handback)
    engine = Engine(settings, tmp_path / "settings.json",
                    [sys.executable, FAKE, "--scenario", scenario, *extra],
                    ddp_port=fake.ddp_port, http_port=fake.http_port, discover=lambda timeout=3.0: [])
    if start:
        engine.start()
    return engine


def test_end_to_end_frames_reach_controller(tmp_path, fake):
    engine = make_engine(tmp_path, fake)
    try:
        assert wait_for(lambda: fake.packet_count() >= 100, 3.0)
        with fake.lock:
            assert all(len(p) == 10 + 234 for _, p in fake.packets[:100])
        status = engine.status()
        assert status["capture"]["state"] == "running" and status["controller"]["state"] == "online"
        assert status["controller"]["pixels"] == 78 and status["controller"]["fps"] == 77
        preview = engine.preview()
        assert preview is not None and len(preview["pixels"]) == 234 and set(preview["meters"]) == {
            "kick", "snare", "hats", "bass", "vocal"}
        json.dumps(status)
    finally:
        engine.stop()


def test_airpods_delay_applied(tmp_path, fake):
    engine = make_engine(tmp_path, fake, "music", "--latency-ms", "170.7")
    try:
        time.sleep(2.5)
        status = engine.status()
        assert 130 <= status["sync"]["delay_ms"] <= 160
        assert abs(status["output"]["latency_ms"] - 160.7) <= 10
        assert status["capture"]["output"]["latency_ms"] == 170.7
    finally:
        engine.stop()


def test_output_switch_mid_stream(tmp_path, fake):
    engine = make_engine(tmp_path, fake, "switch")
    try:
        assert wait_for(lambda: engine.status()["output"]["sent"] > 0, 3.0)
        counts = []
        for _ in range(30):
            counts.append(engine.status()["output"]["sent"])
            time.sleep(.1)
        assert all(b > a for a, b in zip(counts, counts[1:]))
        status = engine.status()
        assert status["capture"]["output"]["name"] == "Fake AirPods"
        assert 130 <= status["sync"]["delay_ms"] <= 160
    finally:
        engine.stop()


def test_controller_offline_then_back(tmp_path):
    fake = FakeWled()
    engine = make_engine(tmp_path, fake)
    try:
        assert wait_for(lambda: engine.status()["controller"]["state"] == "offline", 4.0)
        fake.start()
        assert wait_for(lambda: engine.status()["controller"]["state"] == "online", 3.5)
        assert wait_for(lambda: fake.packet_count() > 10, 2.0)
    finally:
        engine.stop()
        fake.stop()


def test_capture_helper_crash_recovers(tmp_path, fake):
    engine = make_engine(tmp_path, fake, "exit-after-3-blocks")
    states = []
    try:
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            states.append(engine.status()["capture"]["state"])
            time.sleep(.02)
        assert "restarting" in states and states.index("restarting") < len(states) - 1
        assert "running" in states[states.index("restarting"):]
        before = fake.packet_count()
        assert wait_for(lambda: fake.packet_count() > before, 1.0)
    finally:
        engine.stop()


def test_silence_hands_back_and_resumes(tmp_path, fake):
    engine = make_engine(tmp_path, fake, "silence-then-music", "--silence-s", "4", handback=1.0)
    try:
        assert wait_for(lambda: engine.status()["output"]["handback"], 3.5)
        count = fake.packet_count()
        time.sleep(.5)
        assert fake.packet_count() == count
        assert wait_for(lambda: not engine.status()["output"]["handback"], 3.0)
        resumed = time.monotonic()
        assert wait_for(lambda: fake.packet_count() > count, .2)
        assert time.monotonic() - resumed < .25
    finally:
        engine.stop()


def test_quit_restores_controller(tmp_path, fake):
    engine = make_engine(tmp_path, fake)
    assert wait_for(lambda: engine.status()["controller"]["state"] == "online", 3.0)
    engine.stop()
    engine.stop()
    assert fake.state_posts()[-1] == {"on": True, "bri": 128, "live": False}


def test_set_look(tmp_path, fake):
    engine = make_engine(tmp_path, fake)
    try:
        engine.set_look("sync-check")
        assert engine.status()["look"] == "sync-check"
        assert {"id": "sync-check", "name": "Sync check"} in engine.status()["looks"]
        with pytest.raises(ValueError):
            engine.set_look("nope")
        engine.set_master(.25)
        engine.set_blackout(True)
        assert wait_for(lambda: engine.status()["capture"]["output"] is not None, 3.0)
        engine.set_trim(-20)
        status = engine.status()
        assert (status["master"], status["blackout"], status["sync"]["trim_ms"]) == (.25, True, -20)
    finally:
        engine.stop()
    saved = json.loads((tmp_path / "settings.json").read_text())
    assert saved["look"] == "sync-check" and saved["master"] == .25
    assert saved["trim_ms"] == {"fake-speakers": -20.0}


def test_second_instance_refused(tmp_path):
    (tmp_path / "engine.json").write_text(json.dumps({"url": "http://127.0.0.1:8770/"}))
    with open(tmp_path / "engine.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run([sys.executable, "-m", "wled_dj", "--state-dir", str(tmp_path)],
                                capture_output=True, text=True, timeout=20, cwd=ROOT)
    assert result.returncode == 3
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "event": "already-running", "url": "http://127.0.0.1:8770/"}


def test_status_silence_needs_a_full_second(tmp_path, fake):
    engine = make_engine(tmp_path, fake, "music")
    try:
        assert wait_for(lambda: engine.status()["music"]["bpm"] > 0, 8.0)
        assert not any(engine.status()["music"]["silent"] for _ in range(40) if time.sleep(.025) is None)
    finally:
        engine.stop()


class FlakySocket:
    """Raises a network error nobody listed (host down) a few times, then works."""

    def __init__(self, failures, code):
        self.failures, self.code, self.sent = failures, code, 0

    def sendto(self, packet, address):
        if self.failures > 0:
            self.failures -= 1
            raise OSError(self.code, "network trouble")
        self.sent += 1
        return len(packet)

    def close(self):
        pass


@pytest.mark.parametrize("code", [64, 49, 50])   # EHOSTDOWN, EADDRNOTAVAIL, ENETDOWN
def test_unlisted_network_errors_do_not_kill_sending(tmp_path, fake, code):
    engine = make_engine(tmp_path, fake)
    try:
        assert wait_for(lambda: engine.status()["controller"]["state"] == "online", 3.0)
        flaky = FlakySocket(5, code)
        with engine.lock:
            engine.sock = flaky
        assert wait_for(lambda: flaky.sent > 20, 3.0)
        assert engine.sender._thread.is_alive()
    finally:
        engine.stop()


def test_select_controller_rejects_bad_addresses(tmp_path, fake):
    engine = make_engine(tmp_path, fake, start=False)
    for bad in ("192.168.1.50:80", "bad host", "http://192.168.1.50"):
        with pytest.raises(ValueError):
            engine.select_controller(bad)
    engine.select_controller("192.168.1.99")
    assert engine.settings.controllers[0].host == "192.168.1.99"


def test_supervisor_survives_unexpected_errors(tmp_path, fake):
    from wled_dj.output.wled import Wled
    calls = {"n": 0}

    class Grumpy(Wled):
        def probe(self):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise AttributeError("'list' object has no attribute 'get'")
            return super().probe()

    settings = Settings(controllers=[ControllerConfig(fake.host)])
    engine = Engine(settings, tmp_path / "s.json", [sys.executable, FAKE, "--scenario", "music"],
                    ddp_port=fake.ddp_port, http_port=fake.http_port, discover=lambda timeout=3.0: [],
                    wled_factory=Grumpy)
    engine.start()
    try:
        assert wait_for(lambda: engine.status()["controller"]["state"] == "offline", 3.0)
        assert wait_for(lambda: engine.status()["controller"]["state"] == "online", 8.0)
    finally:
        engine.stop()


def test_thread_crash_is_logged_and_stops_the_engine(tmp_path, fake, caplog):
    import threading
    from wled_dj.__main__ import install_crash_hooks
    engine = make_engine(tmp_path, fake, start=False)
    previous = threading.excepthook
    install_crash_hooks(engine)
    try:
        def boom():
            raise RuntimeError("worker exploded")
        thread = threading.Thread(target=boom)
        thread.start()
        thread.join()
        assert engine.crashed and engine.quitting.is_set()
        assert "worker exploded" in caplog.text and "Traceback" in caplog.text
    finally:
        threading.excepthook = previous


def test_probe_failure_mid_show_keeps_original_state(tmp_path):
    fake = FakeWled()
    fake.state.update(on=True, bri=90)
    fake.auto_live = True
    fake.start()
    engine = make_engine(tmp_path, fake, start=False)
    engine.reprobe_s = .5
    engine.start()
    try:
        assert wait_for(lambda: engine.status()["controller"]["state"] == "online", 3.0)
        assert wait_for(lambda: fake.packet_count() > 20, 2.0)
        fake.stop()
        assert wait_for(lambda: engine.status()["controller"]["state"] == "offline", 3.0)
        fake.start()
        states = []
        assert wait_for(lambda: states.append(engine.status()["controller"]["state"]) or states[-1] == "online", 6.0)
        assert "busy" not in states
    finally:
        engine.stop()
        fake.stop()
    assert fake.state_posts()[-1] == {"on": True, "bri": 90, "live": False}


def test_handback_restores_and_takeback_prepares_again(tmp_path):
    fake = FakeWled()
    fake.state.update(on=False, bri=60)
    fake.start()
    engine = make_engine(tmp_path, fake, "silence-then-music", "--silence-s", "4", handback=1.0)
    try:
        assert wait_for(lambda: {"on": False, "bri": 60, "live": False} in fake.state_posts(), 5.0)
        assert wait_for(lambda: fake.state_posts()[-1] == {"on": True, "bri": 255, "lor": 0}, 6.0)
    finally:
        engine.stop()
        fake.stop()
    assert fake.state_posts()[-1] == {"on": False, "bri": 60, "live": False}


def test_controller_address_is_resolved_again_on_reprobe(tmp_path, fake, monkeypatch):
    import wled_dj.service as service
    answers = iter(["127.0.0.1"] + ["127.0.0.1"] * 3 + ["127.0.0.3"] * 100)
    monkeypatch.setattr(service.socket, "gethostbyname", lambda host: next(answers))
    engine = make_engine(tmp_path, fake, start=False)
    engine.reprobe_s = .3
    engine.start()
    try:
        assert wait_for(lambda: engine.address is not None and engine.address[0] == "127.0.0.3", 5.0)
    finally:
        engine.stop()


def test_quit_leaves_a_dark_last_frame(tmp_path, fake):
    engine = make_engine(tmp_path, fake, "music", "--latency-ms", "170.7")   # frames queued ~160 ms ahead
    assert wait_for(lambda: fake.packet_count() > 50, 4.0)
    engine.stop()
    time.sleep(.4)
    with fake.lock:
        last = fake.packets[-1][1]
    assert last[10:] == bytes(234)


def test_quit_while_idle_sends_nothing(tmp_path, fake):
    engine = make_engine(tmp_path, fake, "silence-then-music", "--silence-s", "8", handback=1.0)
    assert wait_for(lambda: engine.status()["output"]["handback"], 4.0)
    time.sleep(.3)
    count = fake.packet_count()
    engine.stop()
    time.sleep(.3)
    assert fake.packet_count() == count


def test_a_short_sound_does_not_take_the_lights_back(tmp_path, fake):
    engine = make_engine(tmp_path, fake, "blip", handback=1.0)
    try:
        assert wait_for(lambda: engine.status()["output"]["handback"], 3.0)
        start = time.monotonic()
        idle = []
        while time.monotonic() - start < 2.5:            # spans the 50 ms sound at 3 s
            idle.append(engine.status()["output"]["handback"])
            time.sleep(.02)
        assert all(idle)
        assert wait_for(lambda: not engine.status()["output"]["handback"], 4.0)   # music at 6 s takes them back
    finally:
        engine.stop()


def test_settings_save_while_audio_never_pauses(tmp_path, fake):
    engine = make_engine(tmp_path, fake, "flood")
    try:
        assert wait_for(lambda: engine.status()["capture"]["state"] == "running", 3.0)
        engine.set_master(.3)
        assert wait_for(lambda: (tmp_path / "settings.json").exists()
                        and json.loads((tmp_path / "settings.json").read_text())["master"] == .3, 3.0)
    finally:
        engine.stop()


def test_a_rebooted_controller_is_prepared_again(tmp_path, fake):
    engine = make_engine(tmp_path, fake, start=False)
    engine.reprobe_s = .3
    engine.start()
    try:
        assert wait_for(lambda: engine.status()["controller"]["state"] == "online", 3.0)
        prepared = fake.state_posts().count({"on": True, "bri": 255, "lor": 0})
        with fake.lock:                                   # reboot: uptime restarts, boot brightness, new layout
            fake.info["uptime"] = 3
            fake.state.update(on=True, bri=128)
            fake.cfg["hw"]["led"]["ins"] = [{"start": 0, "len": 78, "rev": True}]
        assert wait_for(lambda: fake.state_posts().count({"on": True, "bri": 255, "lor": 0}) > prepared, 3.0)
        assert wait_for(lambda: engine.status()["controller"]["buses"] == [{"start": 0, "length": 78, "reversed": True}], 3.0)
    finally:
        engine.stop()
    assert fake.state_posts()[-1] == {"on": True, "bri": 128, "live": False}   # still the state from launch


def test_capture_drops_reach_the_status(tmp_path, fake):
    engine = make_engine(tmp_path, fake, "music", "--drops", "7")
    try:
        assert wait_for(lambda: engine.status()["capture"].get("drops") == 7, 3.0)
    finally:
        engine.stop()


def test_output_change_with_a_tap_rebuild_gap(tmp_path, fake):
    engine = make_engine(tmp_path, fake, "rebuild")
    try:
        assert wait_for(lambda: fake.packet_count() > 0, 3.0)
        first = fake.packet_count()
        time.sleep(3.0)                                      # across the 0.4 s gap at about 1 s
        with fake.lock:
            arrivals = [t for t, _ in fake.packets[first - 1:]]
        # The engine promises dark frames once audio has been missing for 0.1 s, so the lights never go a
        # whole gap without a frame. The longest pause is that 0.1 s plus a frame or two.
        assert len(arrivals) > 150 and max(np.diff(arrivals)) < .2
        status = engine.status()
        assert status["capture"]["state"] == "running" and status["capture"]["output"]["name"] == "Fake AirPods"
        assert 130 <= status["sync"]["delay_ms"] <= 160
        assert status["music"]["bpm"] > 0 or wait_for(lambda: engine.status()["music"]["bpm"] > 0, 6.0)
    finally:
        engine.stop()


def test_local_network_blocking_is_explained(tmp_path, fake):
    from wled_dj.output.wled import ControllerOffline, Wled

    class Blocked(Wled):
        def probe(self):
            raise ControllerOffline(f"{self.host}: <urlopen error [Errno 65] No route to host>")

    settings = Settings(controllers=[ControllerConfig("192.168.1.50")])
    engine = Engine(settings, tmp_path / "s.json", [sys.executable, FAKE, "--scenario", "music"],
                    discover=lambda timeout=3.0: [], wled_factory=Blocked)
    engine.start()
    try:
        assert wait_for(lambda: engine.status()["controller"]["state"] == "offline", 3.0)
        assert "Local Network" in engine.status()["controller"]["message"]
    finally:
        engine.stop()


def test_engine_exits_when_the_app_goes_away(tmp_path):
    launcher = (
        "import subprocess, sys\n"
        f"p = subprocess.Popen([sys.executable, '-m', 'wled_dj', '--state-dir', {str(tmp_path)!r}, '--watch-parent',"
        f" '--capture', {sys.executable + ' ' + FAKE + ' --scenario music'!r}],"
        " stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)\n"
        "line = p.stdout.readline()\n"                     # wait for the engine's ready line, then go away
        "print(p.pid if b'ready' in line else -1)\n")
    # Layer separation is off here: with it on, the engine would look for its model and download it.
    (tmp_path / "settings.json").write_text('{"controllers": [{"host": "127.0.0.1"}], "port": 8799, "stems": false}')
    pid = int(subprocess.check_output([sys.executable, "-c", launcher], cwd=ROOT, text=True, timeout=20).strip())
    assert pid > 0, "engine never became ready"
    import os
    try:
        assert wait_for(lambda: not _alive(pid), 8.0)
    finally:
        if _alive(pid):
            os.kill(pid, 9)
    log = (logs_dir() / "engine.log").read_text()            # in the tests' throwaway home, not the real one
    assert "The app that started this engine is gone" in log
    assert "separation" not in log.lower() and "download" not in log.lower()
    assert not (app_support_dir() / "models").exists()


def _alive(pid):
    import os
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()[:1] != "Z"


# ---- layer separation ----------------------------------------------------------------------------

import os  # noqa: E402
import threading  # noqa: E402

from wled_dj.stems.link import StemLink  # noqa: E402

FAKE_STEMS = str(Path(__file__).parent / "fakes" / "fake_stems.py")


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def make_stems_engine(tmp_path, fake, *stems_options, prepare=None, latency="22.1", stems_setting=True, block=512):
    """Capture blocks hold 512 samples, like the real capture's: not one hop each."""
    settings = Settings(controllers=[ControllerConfig(fake.host)], stems=stems_setting)
    link = StemLink([sys.executable, FAKE_STEMS, *stems_options])
    engine = Engine(settings, tmp_path / "settings.json",
                    [sys.executable, FAKE, "--latency-ms", latency, "--block", str(block)],
                    ddp_port=fake.ddp_port, http_port=fake.http_port, discover=lambda timeout=3.0: [],
                    stems=link, prepare_stems=prepare)
    engine.start()
    return engine, link


def layers(engine):
    return engine.status()["layers"]


def test_engine_without_stems_reports_off(tmp_path, fake):
    engine = make_engine(tmp_path, fake)
    try:
        assert wait_for(lambda: engine.preview() is not None and fake.packet_count() >= 20, 3.0)
        assert layers(engine) == {"enabled": False, "state": "off", "source": "bands", "late_share": 0.0, "message": ""}
        meters = engine.preview()["layers"]
        assert set(meters) == {"drums", "bass", "vocals", "other"}
        assert all(0 <= m["level"] <= 1 and isinstance(m["present"], bool) for m in meters.values())
        json.dumps(engine.status())
    finally:
        engine.stop()


@pytest.mark.parametrize("block", [128, 512])            # a 192 kHz device's blocks, and a 48 kHz device's
def test_engine_uses_stems(tmp_path, fake, block):
    engine, link = make_stems_engine(tmp_path, fake, block=block)
    try:
        assert wait_for(lambda: layers(engine)["state"] == "on" and layers(engine)["source"] == "stems", 5.0)
        assert layers(engine)["enabled"] and layers(engine)["message"] == ""
        before = fake.packet_count()
        assert wait_for(lambda: fake.packet_count() >= before + 50, 2.0)
        assert engine.status()["sync"]["processing_ms"] < 25
        assert engine.preview()["layers"]["drums"]["present"]
    finally:
        engine.stop()
    assert link.state == "off" and link.pid is None


def test_stem_wait_follows_the_light_delay(tmp_path, fake):
    engine = make_engine(tmp_path, fake, start=False)
    engine.sync.set_output(22.1, 0.0)
    assert engine._stem_wait(5.0, arrived=5.0) == pytest.approx(.010)
    engine.sync.set_output(170.7, 5.0)
    assert engine._stem_wait(10.0, arrived=10.0) == pytest.approx(.1)
    engine.sync.set_output(60.0, 10.0)
    assert engine._stem_wait(15.0, arrived=15.0) == pytest.approx(.030)


def test_stem_wait_runs_from_when_the_audio_arrived(tmp_path, fake):
    engine = make_engine(tmp_path, fake, start=False)
    engine.sync.set_output(22.1, 0.0)
    assert engine._stem_wait(5.004, arrived=5.0) == pytest.approx(.006)     # 4 ms in the queue leave 6 of the 10
    assert engine._stem_wait(5.025, arrived=5.0) == pytest.approx(-.015)    # overdue by 15 ms


def test_the_audio_queue_holds_time_not_blocks(tmp_path, fake):
    engine = make_engine(tmp_path, fake, start=False)
    small = np.zeros((128, 2), np.float32)                       # a 192 kHz device's block: 2.67 ms
    for n in range(225):                                         # 0.6 s of audio
        engine._on_audio(AudioBlock(small, n * 128 / 48000, 0))
    assert engine.status()["output"]["audio_dropped"] == 0       # 64 blocks would have kept 171 ms of it
    for n in range(225, 300):                                    # 0.8 s in all, 0.16 s more than it holds
        engine._on_audio(AudioBlock(small, n * 128 / 48000, 0))
    assert engine.status()["output"]["audio_dropped"] == 60      # the oldest 0.16 s went: 60 blocks of 2.67 ms
    block, _ = engine.audio.get(0)
    assert block.host_time == pytest.approx(60 * 128 / 48000)    # what is left starts where the lost audio ended


def test_the_audio_queue_keeps_the_newest_block_however_big(tmp_path, fake):
    engine = make_engine(tmp_path, fake, start=False)
    for n in range(2):
        engine._on_audio(AudioBlock(np.zeros((48000, 2), np.float32), float(n), 0))     # one second each
    assert engine.status()["output"]["audio_dropped"] == 1
    assert engine.audio.get(0)[0].host_time == 1.0 and engine.audio.get(0) is None


def test_audio_that_waited_in_the_queue_gets_a_shorter_wait(tmp_path, fake):
    now = [100.0]
    engine = Engine(Settings(controllers=[ControllerConfig(fake.host)]), tmp_path / "settings.json",
                    [sys.executable, FAKE], ddp_port=fake.ddp_port, http_port=fake.http_port,
                    discover=lambda timeout=3.0: [], clock=lambda: now[0])
    engine.sync.set_output(22.1, 0.0)
    waits = []
    engine.analyzer.feed = lambda block, wait_s=0.0: waits.append(wait_s) or []
    engine._on_audio(AudioBlock(np.zeros((480, 2), np.float32), 99.99, 0))      # arrives at 100.000
    now[0] = 100.004                                                            # the engine gets to it 4 ms later
    loop = threading.Thread(target=engine._engine_loop, daemon=True)
    loop.start()
    try:
        assert wait_for(lambda: waits, 2.0)
    finally:
        engine.stopping.set()
        loop.join(timeout=2)
    assert waits == [pytest.approx(.006)]


def test_missing_model_is_explained(tmp_path, fake):
    engine, link = make_stems_engine(tmp_path, fake, prepare=lambda: False)
    try:
        assert wait_for(lambda: layers(engine)["state"] == "unavailable", 2.0)
        assert "could not be downloaded" in layers(engine)["message"]
        assert wait_for(lambda: fake.packet_count() >= 50, 3.0)
        assert layers(engine)["source"] == "bands" and link.state == "off"
    finally:
        engine.stop()


def test_downloading_is_shown_then_stems_start(tmp_path, fake):
    done = threading.Event()
    engine, link = make_stems_engine(tmp_path, fake, prepare=lambda: done.wait(10))
    try:
        assert wait_for(lambda: layers(engine)["state"] == "downloading", 2.0)
        assert "32 MB" in layers(engine)["message"] and link.state == "off"
        done.set()
        assert wait_for(lambda: layers(engine)["state"] == "on", 5.0)
    finally:
        done.set()
        engine.stop()


def test_busy_helper_pauses_with_a_message(tmp_path, fake):
    engine, link = make_stems_engine(tmp_path, fake, "--delay-ms", "40")
    try:
        assert wait_for(lambda: layers(engine)["state"] == "paused", 4.0)
        assert layers(engine)["message"].startswith("Layer separation paused: this Mac is busy.")
        before = fake.packet_count()
        assert wait_for(lambda: fake.packet_count() >= before + 50, 2.0)
        assert layers(engine)["source"] == "bands"
    finally:
        engine.stop()


def test_toggle_and_quit_leave_no_helper(tmp_path, fake):
    engine, link = make_stems_engine(tmp_path, fake)
    try:
        assert wait_for(lambda: link.state == "on", 5.0)
        first = link.pid
        engine.set_stems(False)
        assert layers(engine)["state"] == "off" and not layers(engine)["enabled"]
        assert wait_for(lambda: not alive(first), 3.0)
        assert wait_for(lambda: (tmp_path / "settings.json").exists()
                        and json.loads((tmp_path / "settings.json").read_text())["stems"] is False, 3.0)
        engine.set_stems(True)
        assert wait_for(lambda: link.state == "on" and link.pid != first, 5.0)
    finally:
        engine.stop()
    assert link.pid is None
    engine, link = make_stems_engine(tmp_path, fake, "--ready-after-s", "2")
    assert wait_for(lambda: link.state == "starting", 2.0)
    starting = link.pid
    engine.stop()
    assert wait_for(lambda: not alive(starting), 3.0) and link.state == "off"


def test_stems_setting_off_never_starts_the_helper(tmp_path, fake):
    engine, link = make_stems_engine(tmp_path, fake, stems_setting=False)
    try:
        assert wait_for(lambda: fake.packet_count() >= 30, 3.0)
        assert link.state == "off" and layers(engine)["state"] == "off" and not layers(engine)["enabled"]
    finally:
        engine.stop()


# ---- structure in the status ---------------------------------------------------------------------------

def test_status_carries_the_structure(tmp_path, fake):
    from wled_dj.analysis.structure import SECTIONS
    engine = make_engine(tmp_path, fake, start=False)
    assert engine.status()["structure"] == {"section": "quiet", "energy": 0.0, "tension": 0.0, "colour": 0.0, "lead": None}
    engine.start()
    try:
        assert wait_for(lambda: engine.status()["structure"]["energy"] > 0, 5.0)
        structure = engine.status()["structure"]
        assert set(structure) == {"section", "energy", "tension", "colour", "lead"}
        assert structure["section"] in SECTIONS and 0 < structure["energy"] <= 1 and 0 <= structure["tension"] <= 1
        assert 0 <= structure["colour"] < 360 and (structure["lead"] is None or 0 <= structure["lead"] <= 1)
        json.dumps(engine.status())
    finally:
        engine.stop()


def test_the_section_is_read_from_the_stems(tmp_path, fake):
    engine, link = make_stems_engine(tmp_path, fake)
    try:
        assert wait_for(lambda: layers(engine)["source"] == "stems", 5.0)
        assert wait_for(lambda: engine.status()["structure"]["section"] == "groove", 8.0)      # the fake music has drums
    finally:
        engine.stop()


# ---- the flash limit --------------------------------------------------------------------------------------

def test_set_flash_limit(tmp_path, fake):
    engine = make_engine(tmp_path, fake)
    try:
        assert engine.status()["flash_limit"] == 3 and engine.compositor.flash_limit == 3
        engine.set_flash_limit(1)
        assert engine.status()["flash_limit"] == 1 and engine.compositor.flash_limit == 1
        for bad in (4, -1, 1.5, True, "2", None):
            with pytest.raises(ValueError, match="0 \\(off\\) to 3"):
                engine.set_flash_limit(bad)
        assert engine.status()["flash_limit"] == 1
        engine.set_flash_limit(0)
        assert wait_for(lambda: (tmp_path / "settings.json").exists()
                        and json.loads((tmp_path / "settings.json").read_text()).get("flash_limit") == 0, 3.0)
    finally:
        engine.stop()
    from wled_dj.config import load
    saved = load(tmp_path / "settings.json")                      # what the app reads when it starts again
    again = Engine(saved, tmp_path / "settings.json", [sys.executable, FAKE],
                   ddp_port=fake.ddp_port, http_port=fake.http_port, discover=lambda timeout=3.0: [])
    assert saved.flash_limit == 0 and again.status()["flash_limit"] == 0 and again.compositor.flash_limit == 0


def test_a_look_is_told_the_flash_limit(tmp_path, fake):
    """A look that budgets its own flashes carries `flash_limit`; the engine keeps it at the setting."""
    from wled_dj.looks import LOOKS
    from wled_dj.looks.sync_check import SyncCheck

    class Budgeting(SyncCheck):
        id, name, flash_limit = "budgeting", "Budgeting", 3

    LOOKS["budgeting"] = Budgeting
    try:
        engine = make_engine(tmp_path, fake, start=False)
        engine.set_flash_limit(2)
        engine.set_look("budgeting")
        engine.start()
        assert wait_for(lambda: isinstance(engine.look, Budgeting) and engine.look.flash_limit == 2, 3.0)
        engine.set_flash_limit(1)
        assert engine.look.flash_limit == 1
        engine.stop()
    finally:
        del LOOKS["budgeting"]
