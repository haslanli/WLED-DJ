"""The engine: capture -> analysis -> look -> compositor -> timed sender -> WLED, plus control and status."""
from __future__ import annotations

from collections import deque
import copy
import logging
import math
from pathlib import Path
import socket
import threading
import time
from typing import Callable

from .analysis.analyzer import Analyzer
from .analysis.features import Features
from .analysis.frontend import RATE
from .capture import AudioBlock, CaptureClient, CaptureStatus, OutputInfo
from .compositor import Compositor
from .config import ControllerConfig, Settings, save, valid_host
from .looks import DEFAULT, LOOKS, Look
from .output.ddp import DDP_PORT, packets
from .output.discovery import discover as default_discover
from .output.sender import Sender, TimedFrame
from .output.wled import ControllerBusy, ControllerInfo, ControllerOffline, Wled
from .stems.link import StemLink
from .sync import SyncModel

log = logging.getLogger(__name__)

DEFAULT_PIXELS = 78
AUDIO_QUEUE_S = .64       # audio that may wait for the engine; beyond it the oldest is lost
SILENCE_TICK_S = .01
NO_AUDIO_AFTER_S = .1
HANDBACK_FADE_S = 1.0
METER_DECAY_S = .15
SILENT_STATUS_S = 1.0
TAKEBACK_WINDOW_S = .3    # handback ends once at least TAKEBACK_SOUND_S of the last 0.3 s had sound,
TAKEBACK_SOUND_S = .1     # so music takes the lights back at once but a short ping does not     # the studio calls it silence only after a full second of it
RETRY_S = 2.0
SEARCH_RETRY_S = 10.0
REPROBE_S = 10.0
ERROR_STREAK_S = 5.0
LIVE_GRACE_S = 6.0        # WLED reports "live" for a while after our own last frame; that is not "busy"
SAVE_EVERY_S = 1.0
METERS = ("kick", "snare", "hats", "bass", "vocal")
LAYER_METERS = ("drums", "bass", "vocals", "other")
METER_FLOOR_DB = -60.0    # a layer at this level or below shows as an empty meter
# How long the analysis may wait for a hop's stems after its audio arrived: the light delay less a margin,
# within these bounds.
STEM_WAIT_MIN_S, STEM_WAIT_MAX_S, STEM_WAIT_MARGIN_S = .010, .100, .020
LAYER_MESSAGES = {
    "paused": "Layer separation paused: this Mac is busy. The lights follow hits and loudness until it resumes.",
    "downloading": "Getting the layer separation model (32 MB, once).",
    "unavailable": ("Layer separation is unavailable: its model could not be downloaded. "
                    "The lights follow hits and loudness."),
}


class AudioQueue:
    """Captured audio waiting for the engine, each block with the moment it arrived. It holds a length of
    time, not a number of blocks: a device with a small buffer sends many small blocks. When it would hold
    more than `seconds`, the oldest blocks are dropped (never the newest)."""

    def __init__(self, seconds: float):
        self.limit = round(seconds * RATE)                     # samples
        self.dropped = 0                                       # blocks lost because the engine fell behind
        self._items: deque[tuple[AudioBlock, float]] = deque()
        self._samples = 0
        self._arrived = threading.Condition()

    def put(self, block: AudioBlock, arrived: float) -> None:
        with self._arrived:
            self._items.append((block, arrived))
            self._samples += len(block.samples)
            while self._samples > self.limit and len(self._items) > 1:
                self._samples -= len(self._items.popleft()[0].samples)
                self.dropped += 1
            self._arrived.notify()

    def get(self, timeout: float) -> tuple[AudioBlock, float] | None:
        """The oldest block and when it arrived, or None when nothing comes within `timeout`."""
        with self._arrived:
            if not self._arrived.wait_for(lambda: self._items, timeout):
                return None
            item = self._items.popleft()
            self._samples -= len(item[0].samples)
            return item


def offline_message(error: str) -> str:
    if "No route to host" in error or "Errno 65" in error:
        return ("macOS may be blocking WLED DJ from your local network: allow it under System Settings → "
                "Privacy & Security → Local Network, then use Restart Engine.")
    return ""


class Engine:
    def __init__(self, settings: Settings, settings_file: Path, capture_cmd: list[str], *,
                 ddp_port: int = DDP_PORT, http_port: int = 80, clock=time.monotonic, discover=default_discover,
                 wled_factory=Wled, stems: StemLink | None = None,
                 prepare_stems: Callable[[], bool] | None = None):
        self.settings = settings
        self.settings_file = settings_file
        self.ddp_port = ddp_port
        self.http_port = http_port
        self.clock = clock
        self.discover = discover
        self.wled_factory = wled_factory
        self.crashed = False
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.quitting = threading.Event()
        self.wake_supervisor = threading.Event()
        self.sync = SyncModel()
        self.compositor = Compositor()
        self.compositor.master = settings.master
        self.compositor.flash_limit = settings.flash_limit
        self.stems = stems                        # layer separation; None = not available in this run
        self.prepare_stems = prepare_stems        # makes sure the model is on disk (may download it)
        self.stems_wanted = False
        self.stems_note = ""                      # "downloading" or "unavailable" while the link is not started
        self.analyzer = Analyzer(stems)
        self.audio = AudioQueue(AUDIO_QUEUE_S)
        self.capture_drops = 0                    # buffers the helper could not deliver (it counts them)
        self.look_id = settings.look if settings.look in LOOKS else DEFAULT
        self.pixels = DEFAULT_PIXELS
        self.look = self._new_look()
        self.pending_look: str | None = None
        self.capture_status = CaptureStatus("starting")
        self.output: OutputInfo | None = None
        self.controller = {"state": "searching", "host": None, "name": "", "pixels": None, "buses": [],
                           "fps": None, "candidates": [], "message": ""}
        self.wled: Wled | None = None
        self.wleds: dict[str, Wled] = {}          # one per host for the whole run: keeps the user's original state
        self.reprobe_s = REPROBE_S
        self.stream_host: str | None = None
        self.stream_at = -1e9
        self.want_release = False                 # handback: give WLED its own on/off and brightness
        self.want_prepare = False                 # takeback: full brightness for the show again
        self.sock: socket.socket | None = None
        self.address: tuple[str, int] | None = None
        self.sequence = 0
        self.error_since: float | None = None
        self.sender = Sender(self._transmit, fps=60, clock=clock)
        self.sender.active = False
        self.online = False
        self.silent_since: float | None = None
        self.recent_sound: deque[tuple[float, bool]] = deque()
        self.handback = False
        self.meters = {name: 0.0 for name in METERS}
        self.meter_t: float | None = None
        self.last_features: Features | None = None
        self.latest_preview: dict | None = None
        self.dirty = False
        self.saved_at = 0.0
        self.capture = CaptureClient(capture_cmd, self._on_audio, self._on_output, self._on_capture_status)
        self.threads: list[threading.Thread] = []
        self.stopped = False

    # ---- lifecycle -------------------------------------------------------------------------
    def start(self) -> None:
        self.sender.start()
        self.capture.start()
        if self.settings.stems:
            self._want_stems(True)
        for target, name in ((self._engine_loop, "engine"), (self._supervise_controller, "controller")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self.threads.append(thread)

    def stop(self) -> None:
        with self.lock:
            if self.stopped:
                return
            self.stopped = True
        self.stopping.set()
        self.wake_supervisor.set()
        self.capture.stop()
        for thread in self.threads:
            thread.join(timeout=6)
        self._want_stems(False)
        self.sender.stop()                                  # nothing queued may follow the last frame
        if self.online and self.address is not None and not self.handback:
            self._transmit_now(bytes(3 * self.pixels))     # leave a dark frame, not a frozen one
        self._release_controller()
        self._save(force=True)

    def quit(self) -> None:
        self.quitting.set()

    def wait(self) -> None:
        while not (self.quitting.is_set() or self.stopping.is_set()):
            self.quitting.wait(.2)

    # ---- capture callbacks -----------------------------------------------------------------
    def _on_audio(self, block: AudioBlock) -> None:
        self.capture_drops = block.drops
        self.audio.put(block, self.clock())

    def _on_output(self, info: OutputInfo) -> None:
        with self.lock:
            self.output = info
            self.sync.set_output(info.latency_ms, self.clock())
            self.sync.set_trim(self.settings.trim_ms.get(info.uid, 0.0))
        log.info("Output: %s (%s), %.1f ms", info.name, info.transport, info.latency_ms)

    def _on_capture_status(self, status: CaptureStatus) -> None:
        with self.lock:
            self.capture_status = status

    # ---- engine loop -----------------------------------------------------------------------
    def _engine_loop(self) -> None:
        last_audio = self.clock()
        while not self.stopping.is_set():
            item = self.audio.get(SILENCE_TICK_S)
            if item is None:
                now = self.clock()
                if now - last_audio >= NO_AUDIO_AFTER_S:
                    self._render(Features.silence(now), now, real=False)
                    last_audio = now - NO_AUDIO_AFTER_S + SILENCE_TICK_S
                self._save()
                continue
            block, arrived = item
            last_audio = self.clock()
            for features in self.analyzer.feed(block, self._stem_wait(last_audio, arrived)):
                self._render(features, self.clock(), real=True)
            self._save()

    def _render(self, f: Features, now: float, real: bool) -> None:
        with self.lock:
            if self.pending_look is not None or self.look.pixels != self.pixels:
                self.look_id = self.pending_look or self.look_id
                self.look = self._new_look()
                self.pending_look = None
            look = self.look
            lead = self.sync.lead(now)
            offset = self.sync.offset(now)
        rgb = look.render(f, lead)
        self._update_handback(f, now)
        frame = self.compositor.compose(rgb, f.t)
        self.sender.submit(TimedFrame(max(now, f.t + offset), f.t, frame))
        if real:
            self.sync.observe_processing((now - f.t) * 1000)
        self._update_preview(f, self.compositor.preview())

    def _new_look(self) -> Look:
        """The chosen look for the strip as it is, told the flash limit if it budgets its own flashes."""
        look = LOOKS[self.look_id](self.pixels)
        if hasattr(look, "flash_limit"):
            look.flash_limit = self.compositor.flash_limit
        return look

    def _update_handback(self, f: Features, now: float) -> None:
        with self.lock:
            self.recent_sound.append((f.t, not f.silent))
            while self.recent_sound and f.t - self.recent_sound[0][0] > TAKEBACK_WINDOW_S:
                self.recent_sound.popleft()
            if f.silent:
                if self.silent_since is None:
                    self.silent_since = now
                quiet = now - self.silent_since - self.settings.handback_after_s
                if not self.handback:
                    self.compositor.fade = 1.0 if quiet <= 0 else max(0.0, 1 - quiet / HANDBACK_FADE_S)
                if quiet >= HANDBACK_FADE_S and not self.handback:
                    self.handback = True
                    self.want_release, self.want_prepare = True, False
                    log.info("Silence: handing the lights back to WLED")
            elif not self.handback:
                self.silent_since = None
                self.compositor.fade = 1.0
            elif sum(sound for _, sound in self.recent_sound) * .01 >= TAKEBACK_SOUND_S - 1e-9:
                self.handback = False
                self.silent_since = None
                self.compositor.fade = 1.0
                self.want_prepare = True
                log.info("Music again: taking the lights back")
            self.sender.active = self.online and not self.handback

    def _update_preview(self, f: Features, frame) -> None:
        with self.lock:
            dt = 0.0 if self.meter_t is None else max(0.0, f.t - self.meter_t)
            self.meter_t = f.t
            decay = math.exp(-dt / METER_DECAY_S)
            for name in METERS:
                onset = getattr(f, name)
                self.meters[name] = max(self.meters[name] * decay, onset.strength if onset else 0.0)
            self.last_features = f
            self.latest_preview = {"pixels": frame.tobytes(), "meters": dict(self.meters), "beat": f.beat.phase,
                                   "layers": self._layer_meters(f)}

    # ---- layer separation ------------------------------------------------------------------
    def _stem_wait(self, now: float, arrived: float) -> float:
        """How long the analysis may still wait for the stems of audio that reached the engine at `arrived`:
        as long as the light delay leaves room for, counted from that moment. Negative once that has passed."""
        allowed = min(STEM_WAIT_MAX_S, max(STEM_WAIT_MIN_S, self.sync.offset(now) - STEM_WAIT_MARGIN_S))
        return arrived + allowed - now

    def _want_stems(self, on: bool) -> None:
        if self.stems is None:
            return
        with self.lock:
            self.stems_wanted = on
            self.stems_note = ""
        if not on:
            self.stems.stop()
        elif self.prepare_stems is None:
            self.stems.start()
        else:
            threading.Thread(target=self._prepare_and_start_stems, name="stems-setup", daemon=True).start()

    def _prepare_and_start_stems(self) -> None:
        with self.lock:
            self.stems_note = "downloading"
        try:
            ready = bool(self.prepare_stems())
        except Exception:
            log.exception("Preparing layer separation failed")
            ready = False
        with self.lock:
            if not self.stems_wanted:               # switched off or quitting while the model was fetched
                return
            self.stems_note = "" if ready else "unavailable"
            if ready:
                self.stems.start()

    @staticmethod
    def _layer_meters(f: Features) -> dict:
        if f.layers is None:
            return {name: {"level": 0.0, "present": False} for name in LAYER_METERS}
        return {name: {"level": round(min(1.0, max(0.0, 1 - f.layers[name].level_db / METER_FLOOR_DB)), 3),
                       "present": f.layers[name].present} for name in LAYER_METERS}

    @staticmethod
    def _structure_status(f: Features | None) -> dict:
        """Where the song is, for the studio: section, energy and tension (0-1), the song's colour in degrees on
        the circle of fifths, and the lead's place in its range (None while no lead sounds)."""
        if f is None or f.structure is None or f.tone is None or f.voice is None:
            return {"section": "quiet", "energy": 0.0, "tension": 0.0, "colour": 0.0, "lead": None}
        return {"section": f.structure.section, "energy": round(f.structure.energy, 2),
                "tension": round(f.structure.tension, 2), "colour": round(f.tone.angle, 1) % 360,
                "lead": round(f.voice.position, 2) if f.voice.level > 0 else None}

    def _layers_status(self, f: Features | None) -> dict:
        enabled = self.stems is not None and self.settings.stems
        state = "off" if not enabled else self.stems_note or self.stems.state
        return {"enabled": enabled, "state": state, "source": f.source if f else "bands",
                "late_share": round(self.stems.late_share, 3) if enabled else 0.0,
                "message": LAYER_MESSAGES.get(state, "")}

    # ---- output ----------------------------------------------------------------------------
    def _transmit(self, frame) -> bool:
        with self.lock:
            sock, address = self.sock, self.address
            self.sequence = self.sequence % 15 + 1
            sequence = self.sequence
        if sock is None or address is None:
            return False
        try:
            for packet in packets(frame, sequence):
                sock.sendto(packet, address)
        except OSError as error:
            # Any network error (buffer full, host down, address gone, network down...) is a failed frame:
            # the sender backs off, and a streak of them marks the controller offline. Never fatal.
            with self.lock:
                if self.error_since is None:
                    self.error_since = self.clock()
                    log.warning("Sending to %s failed: %s", address[0], error)
            return False
        with self.lock:
            self.error_since = None
            self.stream_host = self.controller.get("host")
            self.stream_at = self.clock()
        return True

    def _transmit_now(self, data: bytes) -> None:
        import numpy as np
        try:
            self._transmit(np.frombuffer(data, np.uint8).reshape(-1, 3))
        except OSError:
            pass

    def _bind(self, host: str, info: ControllerInfo, wled: Wled) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        with self.lock:
            self.wled = wled
            self.sock = sock
            self.address = (socket.gethostbyname(host), self.ddp_port)
            self.error_since = None
            self.pixels = info.pixels
            self.sender.fps = 77 if info.fast_ddp else 60
            self.compositor.gamma = 1.0 if info.wled_gamma else 2.2
            self.online = True
            self.controller.update(state="online", host=host, name=info.name, pixels=info.pixels, message="",
                                   buses=[{"start": b.start, "length": b.length, "reversed": b.reversed}
                                          for b in info.buses], fps=self.sender.fps, candidates=[])
            self.sender.active = not self.handback
        log.info("Controller %s online: %s, %d pixels, %d FPS", host, info.name, info.pixels, self.sender.fps)

    def _unbind(self, state: str) -> None:
        with self.lock:
            self.online = False
            self.sender.active = False
            if self.sock is not None:
                self.sock.close()
            self.sock = None
            self.address = None
            self.controller["state"] = state

    def _release_controller(self) -> None:
        with self.lock:
            wled = self.wled
            self.wled = None
        self._unbind(self.controller["state"])
        if wled is not None:
            try:
                wled.restore()
            except ControllerOffline as error:
                log.warning("Could not restore controller: %s", error)

    # ---- controller supervisor -------------------------------------------------------------
    def _wait(self, seconds: float) -> bool:
        """Sleep, waking early on stop or a controller change. Returns False when stopping."""
        self.wake_supervisor.wait(seconds)
        self.wake_supervisor.clear()
        return not self.stopping.is_set()

    def _chosen_host(self) -> str | None:
        with self.lock:
            return self.settings.controllers[0].host if self.settings.controllers else None

    def _supervise_controller(self) -> None:
        while not self.stopping.is_set():
            try:
                if not self._supervise_once():
                    return
            except Exception as error:               # never let reconnection die; treat it as offline
                log.exception("Controller supervisor error")
                with self.lock:
                    self._unbind("offline")
                    self.controller["message"] = str(error)
                if not self._wait(RETRY_S):
                    return

    def _supervise_once(self) -> bool:
        """One connect-and-watch cycle. Returns False when the engine is stopping."""
        host = self._chosen_host()
        if host is None:
            with self.lock:
                self.controller.update(state="searching", candidates=[])
            found = [(n, h) for n, h in self.discover(3.0) if valid_host(h)]
            if self.stopping.is_set():
                return False
            if len(found) == 1:
                self.select_controller(found[0][1], found[0][0])
                return True
            with self.lock:
                self.controller.update(state="choose" if found else "searching",
                                       candidates=[{"name": n, "host": h} for n, h in found])
            return self._wait(SEARCH_RETRY_S)
        with self.lock:
            wled = self.wleds.get(host)
            if wled is None:
                wled = self.wleds[host] = self.wled_factory(host, http_port=self.http_port)
            ours = self.stream_host == host and self.clock() - self.stream_at < LIVE_GRACE_S
        try:
            info = wled.probe()
            wled.prepare(ignore_live=ours)
        except ControllerBusy as error:
            log.info("%s", error)
            with self.lock:
                self.controller.update(state="busy", host=host)
            return self._wait(RETRY_S)
        except ControllerOffline as error:
            log.info("Controller offline: %s", error)
            with self.lock:
                self.controller.update(state="offline", host=host, message=offline_message(str(error)))
            return self._wait(RETRY_S)
        self._bind(host, info, wled)
        self._watch(host, wled, info)
        return not self.stopping.is_set()

    def _watch(self, host: str, wled: Wled, info: ControllerInfo) -> None:
        next_probe = self.clock() + self.reprobe_s
        while not self.stopping.is_set():
            self.wake_supervisor.wait(.25)
            self.wake_supervisor.clear()
            if self.stopping.is_set():
                return
            if self._chosen_host() != host:
                self._release_controller()
                return
            with self.lock:
                failing = self.error_since is not None and self.clock() - self.error_since >= ERROR_STREAK_S
            if failing:
                log.warning("Controller %s stopped accepting frames", host)
                self._unbind("offline")
                return
            with self.lock:
                release, prepare = self.want_release, self.want_prepare
                self.want_release = self.want_prepare = False
            try:
                if release:
                    wled.release()
                if prepare:
                    wled.prepare(ignore_live=True)
            except ControllerOffline as error:
                log.warning("Controller lost: %s", error)
                self._unbind("offline")
                return
            if self.clock() >= next_probe:
                next_probe = self.clock() + self.reprobe_s
                try:
                    latest = wled.probe()
                    address = (socket.gethostbyname(host), self.ddp_port)   # a .local name may move
                except (ControllerOffline, OSError) as error:
                    log.warning("Controller lost: %s", error)
                    self._unbind("offline")
                    return
                rebooted = latest.uptime < info.uptime
                info = latest
                with self.lock:
                    if address != self.address:
                        log.info("Controller %s moved to %s", host, address[0])
                        self.address = address
                    self.pixels = info.pixels
                    self.controller.update(pixels=info.pixels, buses=[
                        {"start": b.start, "length": b.length, "reversed": b.reversed} for b in info.buses])
                    if rebooted and not self.handback:
                        log.info("Controller %s restarted; putting it back into DJ mode", host)
                        self.want_prepare = True

    # ---- control ---------------------------------------------------------------------------
    def set_look(self, look_id: str) -> None:
        if look_id not in LOOKS:
            raise ValueError(f"Unknown look: {look_id}")
        with self.lock:
            self.pending_look = look_id
            self.look_id = look_id
            self.settings.look = look_id
            self.dirty = True

    def set_master(self, value: float) -> None:
        value = min(1.0, max(0.0, float(value)))
        with self.lock:
            self.compositor.master = value
            self.settings.master = value
            self.dirty = True

    def set_blackout(self, on: bool) -> None:
        with self.lock:
            self.compositor.blackout = bool(on)

    def set_stems(self, on: bool) -> None:
        with self.lock:
            self.settings.stems = bool(on)
            self.dirty = True
        self._want_stems(bool(on))

    def set_flash_limit(self, limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 3:
            raise ValueError("The flash limit is 0 (off) to 3 a second")
        with self.lock:
            self.settings.flash_limit = limit
            self.compositor.flash_limit = limit
            if hasattr(self.look, "flash_limit"):
                self.look.flash_limit = limit
            self.dirty = True

    def set_trim(self, ms: float) -> None:
        with self.lock:
            self.sync.set_trim(ms)
            if self.output is not None:
                self.settings.trim_ms[self.output.uid] = self.sync.trim_ms
                self.dirty = True

    def select_controller(self, host: str, name: str = "") -> None:
        host = host.strip() if isinstance(host, str) else ""
        if not valid_host(host):
            raise ValueError("Enter the controller's IP address or host name, for example 192.168.1.50 (no port)")
        with self.lock:
            others = [c for c in self.settings.controllers if c.host != host]
            self.settings.controllers = [ControllerConfig(host, name)] + others
            self.dirty = True
        self.wake_supervisor.set()

    def _save(self, force: bool = False) -> None:
        with self.lock:
            if not self.dirty or (not force and self.clock() - self.saved_at < SAVE_EVERY_S):
                return
            self.dirty = False
            self.saved_at = self.clock()
            settings = copy.deepcopy(self.settings)      # the studio thread may change it while we write
        try:
            save(settings, self.settings_file)
        except OSError as error:
            log.warning("Could not save settings: %s", error)

    # ---- status ----------------------------------------------------------------------------
    def status(self) -> dict:
        now = self.clock()
        with self.lock:
            capture = self.capture_status
            output = self.output
            f = self.last_features
            result = {
                "capture": {"state": capture.state, "message": capture.message, "drops": self.capture_drops,
                            "output": None if output is None else {"name": output.name, "uid": output.uid,
                                                                    "latency_ms": output.latency_ms}},
                "sync": {"delay_ms": round(self.sync.delay(now) * 1000, 1),
                         "lead_ms": round(self.sync.lead(now) * 1000, 1),
                         "trim_ms": self.sync.trim_ms, "processing_ms": round(self.sync.processing_ms, 1)},
                "controller": {k: (list(v) if isinstance(v, list) else v) for k, v in self.controller.items()},
                "look": self.look_id,
                "looks": [{"id": key, "name": cls.name} for key, cls in LOOKS.items()],
                "master": self.compositor.master,
                "blackout": self.compositor.blackout,
                "music": {"bpm": round(f.beat.bpm, 1) if f else 0.0,
                          "confidence": round(f.beat.confidence, 2) if f else 0.0,
                          "silent": self.handback or (self.silent_since is not None
                                                      and now - self.silent_since >= SILENT_STATUS_S)},
                "layers": self._layers_status(f),
                "structure": self._structure_status(f),
                "flash_limit": self.compositor.flash_limit,
            }
            handback = self.handback
        stats = self.sender.stats()
        result["output"] = {**{k: (round(v, 1) if isinstance(v, float) else v) for k, v in stats.items()},
                            "handback": handback, "audio_dropped": self.audio.dropped}
        return result

    def preview(self) -> dict | None:
        with self.lock:
            return None if self.latest_preview is None else dict(self.latest_preview)
