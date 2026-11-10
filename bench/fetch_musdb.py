"""Fetch the MUSDB18 7-second sample set (research use only, kept out of git) into bench/data/musdb7.

Usage: uv run python bench/fetch_musdb.py [--zip LOCAL_ZIP]
"""
import argparse
from pathlib import Path
import shutil
import urllib.request
import zipfile

URL = "https://github.com/sigsep/sigsep-mus-db/releases/download/v0.4.0/MUSDB18-7-STEMS.zip"
DATA = Path(__file__).resolve().parent / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, help="use an already downloaded MUSDB18-7-STEMS.zip")
    args = parser.parse_args()
    target = DATA / "musdb7"
    if (target / "test").is_dir() and (target / "train").is_dir():
        print(f"already present: {target}")
    else:
        DATA.mkdir(parents=True, exist_ok=True)
        archive = DATA / "MUSDB18-7-STEMS.zip"
        if args.zip:
            shutil.copyfile(args.zip, archive)
        elif not archive.exists():
            print(f"downloading {URL} (~147 MB)")
            urllib.request.urlretrieve(URL, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
    counts = {split: len(list((target / split).glob("*.stem.mp4"))) for split in ("test", "train")}
    if counts != {"test": 50, "train": 94}:
        raise SystemExit(f"unexpected clip counts {counts}")
    print(f"ok: {counts} in {target}")


if __name__ == "__main__":
    main()
