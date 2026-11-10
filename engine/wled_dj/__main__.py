"""Run the engine: `python -m wled_dj [--capture CMD] [--settings PATH] [--state-dir DIR] [--port N] [--open]`.

stdout carries one JSON object per line for the menu-bar app: ready / already-running."""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
from pathlib import Path
import shlex
import signal
import sys
import threading

from .config import app_support_dir, load, logs_dir
from .logs import setup_logging
from .looks import load_looks
from .service import Engine
from .stems.link import StemLink
from .stems.models import ensure_models, models_dir
from .studio import serve

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTURE = ROOT / "app" / ".build" / "release" / "wled-capture"


log = logging.getLogger("wled_dj")


def emit(event: dict) -> None:
    print(json.dumps(event), flush=True)


def install_crash_hooks(engine) -> None:
    """Any uncaught exception, in any thread, is logged with its traceback and stops the engine with an
    error code, so the menu-bar app restarts it instead of leaving a half-dead engine running."""
    def crashed(kind, exc_type, exc, tb) -> None:
        log.critical("Uncaught exception in %s", kind, exc_info=(exc_type, exc, tb))
        engine.crashed = True
        engine.quit()

    sys.excepthook = lambda t, e, tb: crashed("main thread", t, e, tb)
    threading.excepthook = lambda args: crashed(f"thread {args.thread.name if args.thread else '?'}",
                                                args.exc_type, args.exc_value, args.exc_traceback)


def watch_parent(engine, interval: float = 1.0) -> None:
    """When the app is force-quit, its engine is re-parented; quit instead of running on unseen."""
    parent = os.getppid()

    def watch() -> None:
        while not engine.quitting.wait(interval):
            if os.getppid() != parent:
                log.warning("The app that started this engine is gone; quitting")
                engine.quit()
                return

    threading.Thread(target=watch, name="parent-watchdog", daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wled_dj")
    parser.add_argument("--capture", help="capture helper command (shell-style)")
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--port", type=int)
    parser.add_argument("--open", action="store_true", help="open the studio in the browser")
    parser.add_argument("--looks", type=Path, default=None,
                        help="folder of look files to load (default: the looks folder beside the settings)")
    parser.add_argument("--watch-parent", action="store_true",
                        help="quit when the launching app goes away (used by WLED DJ.app)")
    args = parser.parse_args(argv)
    state_dir = args.state_dir or app_support_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = open(state_dir / "engine.lock", "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            url = json.loads((state_dir / "engine.json").read_text()).get("url")
        except (OSError, ValueError):
            url = None
        emit({"event": "already-running", "url": url})
        return 3
    setup_logging(logs_dir())
    settings_file = args.settings or state_dir / "settings.json"
    looks = load_looks(args.looks or state_dir / "looks")           # before the settings: a saved look may be one of them
    logging.getLogger(__name__).info("Looks loaded: %s", ", ".join(looks) or "none")
    settings = load(settings_file)
    if args.port:
        settings.port = args.port
    capture = args.capture or os.environ.get("WLED_DJ_CAPTURE") or str(DEFAULT_CAPTURE)
    stems = StemLink([sys.executable, "-m", "wled_dj.stems", str(models_dir())])
    engine = Engine(settings, settings_file, shlex.split(capture), stems=stems,
                    prepare_stems=lambda: ensure_models(models_dir()))
    install_crash_hooks(engine)
    if args.watch_parent:
        watch_parent(engine)
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda *_: engine.quit())
    engine.start()
    try:
        url = serve(engine, settings.port)
        temporary = state_dir / "engine.json.tmp"
        temporary.write_text(json.dumps({"url": url, "pid": os.getpid()}))
        os.replace(temporary, state_dir / "engine.json")
        emit({"event": "ready", "url": url})
        if args.open:
            import webbrowser
            webbrowser.open(url)
        engine.wait()
    finally:
        engine.stop()
    return 1 if engine.crashed else 0


if __name__ == "__main__":
    sys.exit(main())
