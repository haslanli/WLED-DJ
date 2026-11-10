"""Frames between the engine and the stem helper: fixed size, little-endian, over the helper's stdin/stdout.

In:  int64 index + 2 x 128 float32 (left, then right) of 44.1 kHz audio.
Out: int64 index + 3 x 128 float32 (drums, bass, vocals), mono.
The helper writes one READY frame when it can work. After that every input frame k is answered by one
output frame k that holds the stems of input k - 1 (zeros for k = 0): the model answers one hop behind."""
from __future__ import annotations

import struct

import numpy as np

HOP, RATE = 128, 44100
LATENCY = 2 * HOP                 # samples a stem trails the audio it was made from
READY = -1
_INDEX = struct.Struct("<q")
IN_BYTES = _INDEX.size + 2 * HOP * 4
OUT_BYTES = _INDEX.size + 3 * HOP * 4


def pack_in(index: int, chunk: np.ndarray) -> bytes:
    return _INDEX.pack(index) + np.ascontiguousarray(chunk, "<f4").tobytes()


def unpack_in(data: bytes) -> tuple[int, np.ndarray]:
    return _INDEX.unpack_from(data)[0], np.frombuffer(data, "<f4", offset=_INDEX.size).reshape(2, HOP)


def pack_out(index: int, stems: np.ndarray) -> bytes:
    return _INDEX.pack(index) + np.ascontiguousarray(stems, "<f4").tobytes()


def unpack_out(data: bytes) -> tuple[int, np.ndarray]:
    return _INDEX.unpack_from(data)[0], np.frombuffer(data, "<f4", offset=_INDEX.size).reshape(3, HOP)


def read_exact(stream, count: int) -> bytes | None:
    """`count` bytes, or None when the stream ends first."""
    data = bytearray()
    while len(data) < count:
        part = stream.read(count - len(data))
        if not part:
            return None
        data.extend(part)
    return bytes(data)
