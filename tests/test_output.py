import socket
import struct
import time

import numpy as np
import pytest

from fakes.fake_wled import FakeWled
from wled_dj.output.ddp import packets
from wled_dj.output.discovery import discover
from wled_dj.output.wled import Bus, ControllerBusy, ControllerOffline, Wled


@pytest.fixture
def fake():
    f = FakeWled().start()
    yield f
    f.stop()


def header(packet):
    return struct.unpack("!BBBBIH", packet[:10])


def test_packets_small_frame():
    frame = np.arange(78 * 3, dtype=np.uint8).reshape(78, 3)
    out = packets(frame, 5)
    assert len(out) == 1 and len(out[0]) == 10 + 234
    assert header(out[0]) == (0x41, 5, 0x0B, 1, 0, 234)
    assert out[0][10:] == frame.tobytes()


def test_packets_split():
    out = packets(np.zeros((1000, 3), np.uint8), 1)
    assert [header(p)[4] for p in out] == [0, 1440, 2880]
    assert [header(p)[0] for p in out] == [0x40, 0x40, 0x41]
    assert [header(p)[5] for p in out] == [1440, 1440, 120]


def test_packets_validate():
    with pytest.raises(ValueError):
        packets(np.zeros((4, 3), np.uint8), 0)
    with pytest.raises(ValueError):
        packets(np.zeros((4, 3), np.uint8), 16)
    with pytest.raises(ValueError):
        packets(np.zeros((4, 3), np.float32), 1)


def test_probe_parses_info(fake):
    ci = Wled(fake.host, http_port=fake.http_port).probe()
    assert (ci.pixels, len(ci.buses), ci.fast_ddp, ci.wled_gamma) == (78, 2, True, True)
    assert ci.buses[0] == Bus(0, 78, False) and ci.name == "Fake WLED" and ci.version == "16.0.1"


def test_probe_reads_gamma_settings(fake):
    fake.cfg["if"]["live"]["no-gc"] = True
    assert Wled(fake.host, http_port=fake.http_port).probe().wled_gamma is False
    fake.cfg["if"]["live"]["no-gc"] = False
    fake.cfg["light"]["gc"]["col"] = 1.0
    fake.info.pop("ddp_fast_refresh")
    ci = Wled(fake.host, http_port=fake.http_port).probe()
    assert ci.wled_gamma is False and ci.fast_ddp is False


def test_prepare_saves_and_restore_restores(fake):
    fake.state.update(on=False, bri=80)
    w = Wled(fake.host, http_port=fake.http_port)
    w.prepare()
    assert fake.state_posts()[-1] == {"on": True, "bri": 255, "lor": 0}
    w.restore()
    assert fake.state_posts()[-1] == {"on": False, "bri": 80, "live": False}


def test_restore_ends_realtime_at_once(fake):
    w = Wled(fake.host, http_port=fake.http_port)
    w.prepare()
    fake.info["live"] = True                     # our own stream is still showing
    start = time.monotonic()
    w.restore()
    assert time.monotonic() - start < .5         # no waiting out WLED's realtime timeout at full brightness
    assert fake.state_posts()[-1] == {"on": True, "bri": 128, "live": False}


def test_prepare_enables_realtime_receive(fake):
    fake.cfg["if"]["live"]["en"] = False
    Wled(fake.host, http_port=fake.http_port).prepare()
    assert ("cfg", {"if": {"live": {"en": True}}}) in fake.posts


def test_busy_controller(fake):
    fake.info["live"] = True
    with pytest.raises(ControllerBusy):
        Wled(fake.host, http_port=fake.http_port).prepare()


def test_offline():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    start = time.monotonic()
    with pytest.raises(ControllerOffline):
        Wled("127.0.0.1", http_port=port, timeout=1.0).probe()
    assert time.monotonic() - start < 1.5


@pytest.mark.live
def test_discover_finds_registered_service():
    from zeroconf import ServiceInfo, Zeroconf
    zc = Zeroconf()
    info = ServiceInfo("_wled._tcp.local.", "Test Strip._wled._tcp.local.",
                       addresses=[socket.inet_aton("127.0.0.1")], port=80, server="test-strip.local.")
    zc.register_service(info)
    try:
        assert ("Test Strip", "127.0.0.1") in discover(3.0)
    finally:
        zc.unregister_service(info)
        zc.close()
