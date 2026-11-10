"""Stand-in for the stem helper: the same pipe protocol, no model.

drums = the previous chunk's mono mean, bass = half of it, vocals = a quarter.
Options: --delay-ms N (before every reply), --stall-after K --stall-ms M (once), --die-after K,
--freeze-after K (stop reading and writing, stay alive), --ready-after-s S."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))
from wled_dj.stems import protocol  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay-ms", type=float, default=0)
    parser.add_argument("--stall-after", type=int, default=-1)
    parser.add_argument("--stall-ms", type=float, default=0)
    parser.add_argument("--die-after", type=int, default=-1)
    parser.add_argument("--freeze-after", type=int, default=-1)
    parser.add_argument("--ready-after-s", type=float, default=0)
    args = parser.parse_args()
    out = sys.stdout.buffer
    time.sleep(args.ready_after_s)
    out.write(protocol.pack_out(protocol.READY, np.zeros((3, protocol.HOP), np.float32)))
    out.flush()
    previous = np.zeros(protocol.HOP, np.float32)
    count = 0
    while True:
        if count == args.freeze_after:
            while True:
                time.sleep(1)
        data = protocol.read_exact(sys.stdin.buffer, protocol.IN_BYTES)
        if data is None:
            return
        index, chunk = protocol.unpack_in(data)
        if count == args.stall_after:
            time.sleep(args.stall_ms / 1000)
        if args.delay_ms:
            time.sleep(args.delay_ms / 1000)
        out.write(protocol.pack_out(index, np.stack((previous, .5 * previous, .25 * previous))))
        out.flush()
        previous = chunk.mean(axis=0)
        count += 1
        if count == args.die_after:
            sys.exit(2)


if __name__ == "__main__":
    try:
        main()
    except (BrokenPipeError, KeyboardInterrupt):
        pass
