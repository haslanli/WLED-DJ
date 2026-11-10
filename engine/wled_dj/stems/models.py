"""The StemgenRT model files: where they come from, where they live, and how they are checked.

The weights are CC BY-NC-SA 4.0 (non-commercial, share alike). They are downloaded once from the
project's own repository at a pinned commit, never shipped with WLED DJ and never passed on."""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Callable
import urllib.request

from ..config import app_support_dir

log = logging.getLogger(__name__)

COMMIT = "7d2d0c19d956f67259b8358c4127f991077e02b0"
BASE_URL = f"https://media.githubusercontent.com/media/sweetspotsoundsystem/stemgen-rt/{COMMIT}/model/parallel/"
# name -> (SHA-256, bytes), as published in the repository's Git LFS pointers
FILES = {"drums.onnx": ("f2a3fc5dd90f45690536370f02d9b9377e6625ab08cb32b3e921371ad134d874", 12267886),
         "bass.onnx": ("1187b414f22344028035caba0a763d2211d13fe8bfce366c5f6b7ee4af77df40", 8660860),
         "vocals.onnx": ("f29a3d8fa596ce4672dd4dc77ccf224f2538389940264aad46ab8168311c2148", 11019332)}
NOTICE = "LICENSE-WEIGHTS.txt"
NOTICE_TEXT = f"""StemgenRT model weights (v0.7 independent models: drums, bass, vocals)
By Axel Delafosse / StemgenRT. Source: https://github.com/sweetspotsoundsystem/stemgen-rt (commit {COMMIT})
Licence: CC BY-NC-SA 4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/
Non-commercial use only. Share alike. These files are unchanged copies, downloaded by WLED DJ for personal use.
Trained on MUSDB18-HQ (Z. Rafii et al., 2019) and MoisesDB (I. Pereira, F. Araujo, F. Korzeniowski, R. Vogl, 2023).
"""


def models_dir() -> Path:
    override = os.environ.get("WLED_DJ_MODELS")
    return Path(override) if override else app_support_dir() / "models"


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def _intact(path: Path, sha256: str, size: int) -> bool:
    try:
        return path.stat().st_size == size and hashlib.sha256(path.read_bytes()).hexdigest() == sha256
    except OSError:
        return False


def verified(directory: Path, files=FILES) -> bool:
    return all(_intact(directory / name, sha256, size) for name, (sha256, size) in files.items())


def ensure_models(directory: Path, files=FILES, fetch: Callable[[str], bytes] = download) -> bool:
    """Make sure every model file is in `directory` and intact, downloading what is missing.
    Returns False (and keeps nothing partial) when a file cannot be fetched or does not match."""
    for name, (sha256, size) in files.items():
        path = directory / name
        if _intact(path, sha256, size):
            continue
        try:
            data = fetch(BASE_URL + name)
        except (OSError, ValueError) as error:
            log.warning("Could not download %s: %s", name, error)
            return False
        if len(data) != size or hashlib.sha256(data).hexdigest() != sha256:
            log.warning("Downloaded %s does not match its published checksum; not keeping it", name)
            path.unlink(missing_ok=True)
            return False
        directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(name + ".part")
        temporary.write_bytes(data)
        os.replace(temporary, path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / NOTICE).write_text(NOTICE_TEXT)
    return True
