"""Runs the built capture helper. Only paths that exit before any audio tap is created (no permission prompt)."""
import os
from pathlib import Path
import subprocess

import pytest

from wled_dj.capture import read_messages

HELPER = Path(__file__).resolve().parents[1] / "app" / ".build" / "release" / "wled-capture"


@pytest.mark.skipif(not HELPER.exists(), reason="helper not built (swift build -c release --package-path app)")
def test_helper_error_reaches_the_engine_before_it_exits():
    for _ in range(5):                      # the bug was a race: repeat
        env = dict(os.environ, WLED_DJ_CAPTURE_SELFTEST="fail")
        process = subprocess.Popen([str(HELPER)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
        events = [m for m in read_messages(process.stdout) if isinstance(m, dict)]
        assert process.wait(timeout=5) == 1
        assert {"event": "error", "message": "Self-test failure"} in events
