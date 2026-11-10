"""Stand-in for the Swift capture helper: speaks the same stdout protocol in real time.

Scenarios: music, silence-then-music, exit-after-3-blocks, no-audio, error-exit, switch.
Blocks hold 512 samples unless --block says otherwise. The real helper sends whatever the audio device's
buffer holds, never exactly one 10 ms hop: 512 frames is usual, which at 192 kHz is 128 samples here.
"""
import argparse
import json
from pathlib import Path
import struct
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RATE = 48000


def emit(kind, payload):
    sys.stdout.buffer.write(struct.pack('<cI', kind, len(payload)) + payload)
    sys.stdout.buffer.flush()


def event(obj):
    emit(b'J', json.dumps(obj).encode())


def output(latency_ms, name='Fake Speakers', uid='fake-speakers'):
    event({'event': 'output', 'name': name, 'uid': uid, 'transport': 'bltn', 'latency_ms': latency_ms})


DROPS = 0


def audio(samples, host_time):
    emit(b'A', struct.pack('<dI', host_time, DROPS) + samples.astype('<f4').tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default='music')
    parser.add_argument('--latency-ms', type=float, default=22.1)
    parser.add_argument('--silence-s', type=float, default=2.5)
    parser.add_argument('--drops', type=int, default=0)
    parser.add_argument('--block', type=int, default=512)
    args = parser.parse_args()
    global DROPS
    DROPS = args.drops
    BLOCK = args.block
    if args.scenario == 'error-exit':
        event({'event': 'error', 'message': 'Fake tap failure 1852797029'})
        sys.exit(1)
    event({'event': 'ready'})
    output(args.latency_ms)
    if args.scenario == 'no-audio':
        while True:
            time.sleep(1)
    import numpy as np
    from signals import music_pattern
    track = music_pattern(60 / 128 * 16)  # four bars, looped
    if args.scenario == 'silence-then-music':
        silent = np.zeros((int(args.silence_s * RATE), 2), np.float32)
        track = np.concatenate((silent, track))
    if args.scenario == 'blip':                      # 6 s of silence with one 50 ms sound at 3 s, then music
        silent = np.zeros((6 * RATE, 2), np.float32)
        silent[3 * RATE:3 * RATE + RATE // 20] = .3
        track = np.concatenate((silent, track))
    start = time.monotonic()
    blocks = 0
    while True:
        index = blocks * BLOCK % (len(track) - BLOCK)
        due = start + blocks * BLOCK / RATE
        delay = due + BLOCK / RATE - time.monotonic()   # a buffer exists only once it is full
        if delay > 0 and args.scenario != 'flood':     # flood: audio always waiting, never a gap
            time.sleep(delay)
        audio(track[index:index + BLOCK], due)
        blocks += 1
        if args.scenario == 'exit-after-3-blocks' and blocks == 3:
            sys.exit(2)
        if args.scenario == 'switch' and blocks == 100:
            output(170.7, "Fake AirPods", 'fake-airpods')
        if args.scenario == 'rebuild' and blocks == 100:   # what the real helper does on an output change:
            time.sleep(.4)                                 # tap torn down and rebuilt, no audio meanwhile,
            event({'event': 'ready'})                      # then a new ready and the new device
            output(170.7, "Fake AirPods", 'fake-airpods')
            start += .4                                    # host time jumps over the gap


if __name__ == '__main__':
    try:
        main()
    except BrokenPipeError:
        pass
