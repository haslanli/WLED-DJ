"""Run the real engine and studio against a fake WLED and the fake capture helper, for looking at the UI.

Usage: uv run python tests/fakes/run_demo.py [--port 8771] [--scenario music] [--latency-ms 170.7]
           [--stems off|fake|real] [--stems-delay-ms N] [--models DIR] [--block 512]

Audio arrives in blocks of --block samples (512, like the real capture on a 48 kHz device; 128 is a 192 kHz one).

--stems real runs the real separation model from --models (default: the app's models folder).
--stems fake runs the fake helper; with --stems-delay-ms it answers late, which pauses separation.
"""
import argparse
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fakes.fake_wled import FakeWled  # noqa: E402
from wled_dj.config import ControllerConfig, Settings  # noqa: E402
from wled_dj.service import Engine  # noqa: E402
from wled_dj.stems.link import StemLink  # noqa: E402
from wled_dj.stems.models import models_dir, verified  # noqa: E402
from wled_dj.studio import serve  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--scenario", default="music")
    parser.add_argument("--latency-ms", default="170.7")
    parser.add_argument("--stems", choices=("off", "fake", "real"), default="off")
    parser.add_argument("--stems-delay-ms", default="0")
    parser.add_argument("--models", type=Path, default=None)
    parser.add_argument("--block", default="512")
    args = parser.parse_args()
    stems = None
    if args.stems == "fake":
        stems = StemLink([sys.executable, str(Path(__file__).with_name("fake_stems.py")), "--delay-ms", args.stems_delay_ms])
    elif args.stems == "real":
        models = args.models or models_dir()
        if not verified(models):
            raise SystemExit(f"no verified model files in {models}")
        stems = StemLink([sys.executable, "-m", "wled_dj.stems", str(models)])
    fake = FakeWled().start()
    fake.info["name"] = "Living room"
    state = Path(tempfile.mkdtemp(prefix="wled-dj-demo-"))
    engine = Engine(Settings(controllers=[ControllerConfig(fake.host)]), state / "settings.json",
                    [sys.executable, str(Path(__file__).with_name("fake_capture.py")), "--scenario", args.scenario,
                     "--latency-ms", args.latency_ms, "--block", args.block],
                    ddp_port=fake.ddp_port, http_port=fake.http_port, discover=lambda timeout=3.0: [], stems=stems)
    engine.start()
    print(serve(engine, args.port), flush=True)
    try:
        engine.wait()
    finally:
        engine.stop()
        fake.stop()


if __name__ == "__main__":
    main()
