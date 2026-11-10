"""Persistent user settings, stored outside the project so they survive rewrites."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
import re

log = logging.getLogger(__name__)

TRIM_LIMIT_MS = 150.0
_LABEL = re.compile(r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)$")


def valid_host(host) -> bool:
    """An IPv4/IPv6 address or a DNS host name (no port, no scheme, no spaces)."""
    if not isinstance(host, str) or not host or len(host) > 253:
        return False
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    if re.fullmatch(r"[\d.]+", host):            # looks numeric but is not a valid address
        return False
    return all(_LABEL.match(label) for label in host.split("."))


@dataclass
class ControllerConfig:
    host: str
    name: str = ""


@dataclass
class Settings:
    controllers: list[ControllerConfig] = field(default_factory=list)
    look: str = "sync-check"
    master: float = 1.0
    trim_ms: dict[str, float] = field(default_factory=dict)
    port: int = 8770
    handback_after_s: float = 8.0
    stems: bool = True          # read the music's layers with the separation model when it keeps up
    flash_limit: int = 3        # full-strip flashes allowed in any one second, whatever the look does; 0 = no limit


def app_support_dir() -> Path:
    return Path.home() / "Library" / "Application Support" / "WLED DJ"


def logs_dir() -> Path:
    return Path.home() / "Library" / "Logs" / "WLED DJ"


def settings_path() -> Path:
    return app_support_dir() / "settings.json"


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _clamp(value, low: float, high: float, default: float) -> float:
    number = _number(value)
    return default if number is None else min(high, max(low, number))


def _parse(raw: dict) -> Settings:
    defaults = Settings()
    controllers = []
    for item in raw.get("controllers", []) if isinstance(raw.get("controllers"), list) else []:
        if isinstance(item, dict) and valid_host(item.get("host")):
            name = item.get("name") if isinstance(item.get("name"), str) else ""
            controllers.append(ControllerConfig(item["host"], name))
    trims = {}
    if isinstance(raw.get("trim_ms"), dict):
        for key, value in raw["trim_ms"].items():
            if isinstance(key, str) and _number(value) is not None:
                trims[key] = _clamp(value, -TRIM_LIMIT_MS, TRIM_LIMIT_MS, 0.0)
    limit = raw.get("flash_limit")
    port = raw.get("port")
    port = port if isinstance(port, int) and not isinstance(port, bool) and 1024 <= port <= 65535 else defaults.port
    return Settings(
        controllers=controllers,
        look=raw["look"] if isinstance(raw.get("look"), str) and raw["look"] else defaults.look,
        master=_clamp(raw.get("master"), 0.0, 1.0, defaults.master),
        trim_ms=trims,
        port=port,
        handback_after_s=_clamp(raw.get("handback_after_s"), 1.0, 120.0, defaults.handback_after_s),
        stems=raw["stems"] if isinstance(raw.get("stems"), bool) else defaults.stems,
        flash_limit=limit if isinstance(limit, int) and not isinstance(limit, bool) and 0 <= limit <= 3 else defaults.flash_limit,
    )


def load(path: Path) -> Settings:
    try:
        text = path.read_text()
    except FileNotFoundError:
        return Settings()
    try:
        raw = json.loads(text)
        if not isinstance(raw, dict):
            raise ValueError("settings must be a JSON object")
    except ValueError as error:
        bad = path.with_name(path.name + ".bad")
        log.warning("Unreadable settings %s (%s); keeping a copy at %s and using defaults", path, error, bad)
        os.replace(path, bad)
        return Settings()
    return _parse(raw)


def save(settings: Settings, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(asdict(settings), indent=1))
    temporary.chmod(0o600)
    os.replace(temporary, path)
