"""DDP (Distributed Display Protocol) packets for RGB frames, as WLED receives them on UDP 4048."""
from __future__ import annotations

import struct

import numpy as np

DDP_PORT = 4048
HEADER = struct.Struct("!BBBBIH")   # flags, sequence, data type, destination, byte offset, length
FLAGS_V1 = 0x40
FLAG_PUSH = 0x01
TYPE_RGB8 = 0x0B
DESTINATION = 1
MAX_PIXELS_PER_PACKET = 480         # 1440 data bytes


def packets(frame: np.ndarray, sequence: int) -> list[bytes]:
    if frame.dtype != np.uint8 or frame.ndim != 2 or frame.shape[1] != 3 or len(frame) == 0:
        raise ValueError("DDP frames are (n, 3) uint8 RGB")
    if not 1 <= sequence <= 15:
        raise ValueError("DDP sequence must be 1..15")
    data = frame.tobytes()
    step = MAX_PIXELS_PER_PACKET * 3
    out = []
    for offset in range(0, len(data), step):
        chunk = data[offset:offset + step]
        flags = FLAGS_V1 | (FLAG_PUSH if offset + step >= len(data) else 0)
        out.append(HEADER.pack(flags, sequence, TYPE_RGB8, DESTINATION, offset, len(chunk)) + chunk)
    return out
