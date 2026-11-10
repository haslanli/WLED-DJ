"""WLED JSON API: what the controller is, and putting it into (and back out of) DJ mode."""
from __future__ import annotations

from dataclasses import dataclass
import json
import urllib.error
import urllib.request


class ControllerOffline(Exception):
    pass


class ControllerBusy(Exception):
    pass


@dataclass(frozen=True)
class Bus:
    start: int
    length: int
    reversed: bool


@dataclass(frozen=True)
class ControllerInfo:
    host: str
    name: str
    version: str
    pixels: int
    buses: list[Bus]
    fast_ddp: bool
    wled_gamma: bool
    live: bool
    uptime: float = 0.0


class Wled:
    def __init__(self, host: str, http_port: int = 80, timeout: float = 2.0):
        self.host = host
        self.base = f"http://{host}" + ("" if http_port == 80 else f":{http_port}")
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.previous: dict | None = None

    def _request(self, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + "/json/" + path, data=data,
                                         headers={"Content-Type": "application/json"} if data else {})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                result = json.load(response)
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise ControllerOffline(f"{self.host}: {error}") from None
        if not isinstance(result, dict):
            raise ControllerOffline(f"{self.host}: not a WLED JSON reply")
        if body is not None and result.get("success") is not True:
            raise ControllerOffline(f"{self.host} rejected {path}: {result}")
        return result

    def probe(self) -> ControllerInfo:
        info = self._request("info")
        cfg = self._request("cfg")
        try:
            pixels = int(info["leds"]["count"])
        except (KeyError, TypeError, ValueError):
            raise ControllerOffline(f"{self.host}: reply has no LED count") from None
        buses = [Bus(int(b.get("start", 0)), int(b.get("len", 0)), bool(b.get("rev", False)))
                 for b in cfg.get("hw", {}).get("led", {}).get("ins", []) if isinstance(b, dict)]
        gc = cfg.get("light", {}).get("gc", {}).get("col", 2.2)
        no_gc = cfg.get("if", {}).get("live", {}).get("no-gc", False)
        return ControllerInfo(self.host, str(info.get("name", "WLED")), str(info.get("ver", "")), pixels,
                              buses or [Bus(0, pixels, False)], info.get("ddp_fast_refresh") is True,
                              bool(gc and float(gc) > 1 and not no_gc), bool(info.get("live", False)),
                              float(info.get("uptime", 0) or 0))

    def prepare(self, ignore_live: bool = False) -> None:
        """Put the controller into DJ mode. The state found the first time is kept for the whole run,
        so reconnecting (or taking the lights back after silence) never overwrites the user's own."""
        if not ignore_live and self._request("info").get("live"):
            raise ControllerBusy(f"{self.host} is already showing another realtime stream")
        if self.previous is None:
            state = self._request("state")
            self.previous = {"on": bool(state.get("on", True)), "bri": int(state.get("bri", 128))}
        if not self._request("cfg").get("if", {}).get("live", {}).get("en", True):
            self._request("cfg", {"if": {"live": {"en": True}}})
        self._request("state", {"on": True, "bri": 255, "lor": 0})

    def release(self) -> None:
        """Give the user's on/off and brightness back, and end realtime mode in the same request so WLED
        never shows its own effect at full brightness in between. The state stays remembered."""
        if self.previous is None:
            return
        self._request("state", {**self.previous, "live": False})

    def restore(self) -> None:
        """Release for good: on quit or when switching to another controller."""
        self.release()
        self.previous = None
