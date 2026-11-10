"""The stem helper process: `python -m wled_dj.stems <models_dir>`.

Reads audio frames on stdin and writes stems on stdout (see protocol.py) until stdin closes.
Lives apart from the engine so the model never shares the engine's interpreter lock."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from . import protocol


def prefer_fast_cores() -> None:
    """Ask macOS to treat the calling thread's work as interactive, which keeps it on the performance cores."""
    try:
        import ctypes
        ctypes.CDLL("/usr/lib/libSystem.B.dylib").pthread_set_qos_class_self_np(0x21, 0)   # user-interactive
    except (OSError, AttributeError):
        pass


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m wled_dj.stems <models_dir>", file=sys.stderr)
        return 1
    try:
        from .runner import StemModel
        model = StemModel(Path(argv[0]), on_thread_start=prefer_fast_cores)
    except Exception as error:               # a missing or damaged file, or no onnxruntime
        print(f"cannot load the stem model from {argv[0]}: {error}", file=sys.stderr)
        return 1
    prefer_fast_cores()
    source, sink = sys.stdin.buffer, sys.stdout.buffer
    sink.write(protocol.pack_out(protocol.READY, np.zeros((3, protocol.HOP), np.float32)))
    sink.flush()
    while True:
        data = protocol.read_exact(source, protocol.IN_BYTES)
        if data is None:
            return 0
        index, chunk = protocol.unpack_in(data)
        sink.write(protocol.pack_out(index, model.step(chunk)))
        sink.flush()


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (BrokenPipeError, KeyboardInterrupt):
        sys.exit(0)
