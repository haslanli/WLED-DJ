import threading

import numpy as np

from wled_dj.stems import runner

LEVEL = {"drums": 1.0, "bass": 2.0, "vocals": 3.0}


def test_the_three_models_run_side_by_side(tmp_path, monkeypatch):
    together = threading.Barrier(3, timeout=2)       # passes only while all three are inside step at once
    ran_on = {}

    class Part:
        def __init__(self, path):
            self.name = path.stem

        def step(self, chunk):
            assert chunk.shape == (1, 2, 128) and chunk.dtype == np.float32
            ran_on[self.name] = threading.get_ident()
            together.wait()
            return np.full(128, LEVEL[self.name], np.float32)

    monkeypatch.setattr(runner, "_Part", Part)
    prepared = []
    model = runner.StemModel(tmp_path, on_thread_start=lambda: prepared.append(threading.get_ident()))
    out = model.step(np.zeros((2, 128), np.float64))
    assert out.shape == (3, 128) and out.dtype == np.float32
    assert [float(row[0]) for row in out] == [1.0, 2.0, 3.0]             # drums, bass, vocals, whoever finishes first
    assert len(set(ran_on.values())) == 3 and threading.get_ident() not in ran_on.values()
    assert sorted(prepared) == sorted(ran_on.values())                   # every model thread was prepared once


def test_the_helper_prepares_every_model_thread_for_the_fast_cores(tmp_path, monkeypatch):
    import io
    import sys
    from types import SimpleNamespace

    from wled_dj.stems import __main__ as helper

    made = {}

    class Model:
        def __init__(self, directory, on_thread_start=None):
            made.update(directory=directory, on_thread_start=on_thread_start)

    monkeypatch.setattr(runner, "StemModel", Model)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"")))        # no audio: ready, then done
    out = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(buffer=out))
    assert helper.main([str(tmp_path)]) == 0
    assert made == {"directory": tmp_path, "on_thread_start": helper.prefer_fast_cores}
    assert len(out.getvalue()) == helper.protocol.OUT_BYTES                           # the one READY frame
