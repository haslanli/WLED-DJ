"""Engine logging: a small rotating file in ~/Library/Logs/WLED DJ plus stderr."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "engine.log"
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3)
    file_handler.setFormatter(formatter)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers[:] = [file_handler, stream]
    # mDNS discovery logs a traceback for every network interface without a route; keep only real errors.
    logging.getLogger("zeroconf").setLevel(logging.ERROR)
    return path
