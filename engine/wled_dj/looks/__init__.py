"""The looks the engine can show.

WLED DJ ships no effects of its own. `LOOKS` starts with one calibration look, Sync check, and `load_looks`
adds yours: every class in the Python files of a folder that has an `id`, a `name` and a `render` method
(see `Look` in base.py, and examples/looks/ for a starting point)."""
from __future__ import annotations

import importlib.util
import inspect
import logging
from pathlib import Path

from .base import Look
from .sync_check import SyncCheck

log = logging.getLogger(__name__)

LOOKS: dict[str, type[Look]] = {"sync-check": SyncCheck}
DEFAULT = "sync-check"


def load_looks(folder: Path) -> list[str]:
    """Register the looks found in `folder` (one or more classes per .py file). Returns the ids added. A file
    that does not import, or a class that is not a look, is logged and left out: one bad file never stops the app."""
    added: list[str] = []
    for path in sorted(Path(folder).glob("*.py")) if Path(folder).is_dir() else []:
        try:
            spec = importlib.util.spec_from_file_location(f"wled_dj_user_look_{path.stem}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as error:                      # anything a user's file can do wrong
            log.warning("Look file %s was not loaded: %s", path.name, error)
            continue
        for _, found in inspect.getmembers(module, inspect.isclass):
            if found.__module__ != module.__name__:
                continue
            look_id, name = getattr(found, "id", None), getattr(found, "name", None)
            if not (isinstance(look_id, str) and look_id and isinstance(name, str) and callable(getattr(found, "render", None))):
                continue
            if look_id in LOOKS:
                log.warning("Look %r in %s was not loaded: that id is taken", look_id, path.name)
                continue
            LOOKS[look_id] = found
            added.append(look_id)
    return added


__all__ = ["DEFAULT", "LOOKS", "Look", "SyncCheck", "load_looks"]
