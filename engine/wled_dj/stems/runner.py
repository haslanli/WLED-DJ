"""StemgenRT as one streaming step: three independent ONNX models (drums, bass, vocals) run side by side.

Each call takes 128 stereo samples at 44.1 kHz and returns the stems of the PREVIOUS 128 samples.
One thread per model. Run one after another, the three need 6 to 7 ms for every 10 ms of audio on a busy
M4, which leaves the engine's 10 ms wait almost no room: one short stall then takes seconds to work off.
Side by side they need about 4 ms, for about half a core more."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

NAMES = ("drums", "bass", "vocals")


class _Part:
    def __init__(self, path: Path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        self.state_names = [i.name for i in inputs[1:]]
        self.outputs = [o.name for o in self.session.get_outputs()]
        self.state = {i.name: np.zeros(i.shape, np.float32) for i in inputs[1:]}

    def step(self, chunk: np.ndarray) -> np.ndarray:
        result = self.session.run(self.outputs, {"audio_chunk": chunk, **self.state})
        self.state = dict(zip(self.state_names, result[1:]))
        return result[0][0, 0].mean(axis=0)


class StemModel:
    def __init__(self, directory: Path, on_thread_start: Callable[[], None] | None = None):
        """on_thread_start runs once on each model's thread before its first step."""
        self.parts = [_Part(Path(directory) / f"{name}.onnx") for name in NAMES]
        self.threads = ThreadPoolExecutor(len(self.parts), thread_name_prefix="stem", initializer=on_thread_start)

    def step(self, chunk: np.ndarray) -> np.ndarray:
        """chunk (2, 128) float32 -> (3, 128) float32 mono: drums, bass, vocals of the previous chunk."""
        feed = np.ascontiguousarray(chunk, np.float32)[None]
        return np.stack(list(self.threads.map(lambda part: part.step(feed), self.parts))).astype(np.float32)
