"""The timeline page for structure, tone and lead (bench/structure_view.py), on made-up features."""
from pathlib import Path
import re
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

from structure_view import page, timeline  # noqa: E402
from wled_dj.analysis.features import Features  # noqa: E402
from wled_dj.analysis.layers import Layer, Layers  # noqa: E402
from wled_dj.analysis.lead import Voice  # noqa: E402
from wled_dj.analysis.structure import Structure  # noqa: E402
from wled_dj.analysis.tone import Tone  # noqa: E402


def scene() -> list[Features]:
    """3 s: a groove, a breakdown from 1 s (a boundary), a drop at 2 s (a boundary and a drop). The lead starts at 1 s."""
    features = []
    for i in range(300):
        t = (i + 1) / 100
        drums = Layer(-12.0, -12.0, -3.0, not 1 <= t < 2, 0.0)
        away = Layer(-70.0, -70.0, -60.0, False, 0.0)
        section = "groove" if t < 1 else "breakdown" if t < 2 else "drop"
        structure = Structure(energy=.6, tension=.2, section=section, since_s=t % 1,
                              boundary={100: .7, 200: 1.0}.get(i, 0.0), drop=i == 200, focus=(.5, .3, .2))
        voice = Voice(position=t / 3, level=.5 if t >= 1 else 0.0, hz=440.0)
        features.append(Features(t=t, layers=Layers(drums, away, away, away, -14.0, "stems"), source="stems",
                                 structure=structure, tone=Tone(90.0, 120.0, .5), voice=voice))
    return features


def test_a_timeline_has_a_player_and_a_mark_for_every_boundary_and_drop():
    markup = timeline(scene(), "Tom & Jerry <live>", "../../corpus/song.wav")
    assert markup.count("<audio") == 1 and 'src="../../corpus/song.wav"' in markup
    assert markup.count('data-mark="boundary"') == 2
    assert markup.count('data-mark="drop"') == 1
    assert "Tom &amp; Jerry &lt;live&gt;" in markup and "<live>" not in markup
    assert re.findall(r'data-section="(\w+)"', markup) == ["groove", "breakdown", "drop"]


def test_the_lead_is_drawn_only_while_it_sounds():
    markup = timeline(scene(), "song", "song.wav")
    lines = re.findall(r'<polyline data-lane="lead" points="([^"]+)"', markup)
    assert len(lines) == 1
    xs = [float(point.split(",")[0]) for point in lines[0].split()]
    assert 1.0 <= xs[0] <= 1.1 and xs[-1] >= 2.9


def test_silence_draws_an_empty_timeline():
    markup = timeline([Features.silence((i + 1) / 100) for i in range(300)], "nothing", "nothing.wav")
    assert markup.count("<audio") == 1 and "data-mark=" not in markup and '<polyline data-lane="lead"' not in markup


def test_the_page_loads_nothing_from_the_internet():
    html = page([timeline(scene(), "song", "song.wav")])
    assert html.startswith("<!doctype html>") and "http" not in html and html.count("<audio") == 1


def test_the_local_server_answers_byte_ranges(tmp_path):
    """A browser seeks in a sound file by asking for a part of it; Python's own file server ignores the request."""
    from threading import Thread
    from urllib.request import Request, urlopen

    from structure_view import server
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "song.wav").write_bytes(bytes(range(256)) * 4)
    httpd = server(tmp_path, 0)
    Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urlopen(Request(base + "/corpus/song.wav", headers={"Range": "bytes=300-349"})) as reply:
            assert reply.status == 206 and reply.headers["Content-Range"] == "bytes 300-349/1024"
            assert reply.read() == (bytes(range(256)) * 4)[300:350]
        with urlopen(Request(base + "/corpus/song.wav", headers={"Range": "bytes=1000-"})) as reply:
            assert reply.status == 206 and len(reply.read()) == 24
        with urlopen(base + "/corpus/song.wav") as reply:
            assert reply.status == 200 and len(reply.read()) == 1024 and reply.headers["Accept-Ranges"] == "bytes"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_the_local_server_answers_only_as_this_computer(tmp_path):
    """Another site's page that points its own name at 127.0.0.1 must not get the files."""
    from threading import Thread
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from structure_view import server
    (tmp_path / "song.wav").write_bytes(b"abc")
    httpd = server(tmp_path, 0)
    Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    try:
        for host in (f"127.0.0.1:{port}", f"localhost:{port}"):
            with urlopen(Request(f"http://127.0.0.1:{port}/song.wav", headers={"Host": host})) as reply:
                assert reply.read() == b"abc"
        with pytest.raises(HTTPError) as refused:
            urlopen(Request(f"http://127.0.0.1:{port}/song.wav", headers={"Host": f"music.example:{port}"}))
        assert refused.value.code == 403
    finally:
        httpd.shutdown()
        httpd.server_close()
