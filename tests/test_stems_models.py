import hashlib
import subprocess
import sys

from wled_dj.config import app_support_dir
from wled_dj.stems import models
from wled_dj.stems.models import BASE_URL, NOTICE, ensure_models, models_dir, verified

GOOD = {"a.onnx": b"alpha" * 100, "b.onnx": b"beta" * 50}
FILES = {name: (hashlib.sha256(data).hexdigest(), len(data)) for name, data in GOOD.items()}


def fetcher(calls, data=GOOD):
    def fetch(url):
        calls.append(url)
        return data[url.rsplit("/", 1)[1]]
    return fetch


def test_downloads_verifies_and_credits(tmp_path):
    calls = []
    target = tmp_path / "models"
    assert not verified(target, FILES)
    assert ensure_models(target, FILES, fetcher(calls))
    assert verified(target, FILES) and (target / "a.onnx").read_bytes() == GOOD["a.onnx"]
    assert calls == [BASE_URL + "a.onnx", BASE_URL + "b.onnx"]
    notice = (target / NOTICE).read_text()
    assert "CC BY-NC-SA 4.0" in notice and "Axel Delafosse" in notice and "MoisesDB" in notice


def test_wrong_content_is_not_kept(tmp_path):
    assert not ensure_models(tmp_path, FILES, fetcher([], {**GOOD, "a.onnx": b"x" * 500}))
    assert not (tmp_path / "a.onnx").exists() and not list(tmp_path.glob("*.part"))
    assert not verified(tmp_path, FILES)


def test_offline_returns_false_and_leaves_nothing_partial(tmp_path):
    def offline(url):
        raise OSError("no route to host")
    assert not ensure_models(tmp_path, FILES, offline)
    assert not list(tmp_path.glob("*.onnx")) and not list(tmp_path.glob("*.part"))


def test_verified_files_are_not_fetched_again(tmp_path):
    assert ensure_models(tmp_path, FILES, fetcher([]))
    calls = []
    assert ensure_models(tmp_path, FILES, fetcher(calls)) and calls == []


def test_a_damaged_file_is_fetched_again(tmp_path):
    assert ensure_models(tmp_path, FILES, fetcher([]))
    (tmp_path / "b.onnx").write_bytes(b"truncated")
    calls = []
    assert not verified(tmp_path, FILES)
    assert ensure_models(tmp_path, FILES, fetcher(calls)) and calls == [BASE_URL + "b.onnx"]


def test_models_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("WLED_DJ_MODELS", raising=False)
    assert models_dir() == app_support_dir() / "models"
    monkeypatch.setenv("WLED_DJ_MODELS", str(tmp_path))
    assert models_dir() == tmp_path


def test_the_pinned_files_are_the_three_models():
    assert set(models.FILES) == {"drums.onnx", "bass.onnx", "vocals.onnx"}
    assert sum(size for _, size in models.FILES.values()) == 31948078
    assert models.COMMIT in BASE_URL and BASE_URL.startswith("https://media.githubusercontent.com/media/sweetspotsoundsystem/stemgen-rt/")


def test_helper_without_models_exits_with_a_message(tmp_path):
    done = subprocess.run([sys.executable, "-m", "wled_dj.stems", str(tmp_path)], stdin=subprocess.DEVNULL,
                          capture_output=True, timeout=30)
    assert done.returncode == 1 and done.stdout == b""
    assert b"stem model" in done.stderr
