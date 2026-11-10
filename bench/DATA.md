# Bench data

Nothing under `bench/corpus/`, `bench/data/` or `bench/out/` is in git. Put your own there.

| Path | What | Where it comes from |
|---|---|---|
| `bench/corpus/` | Music to analyse and render looks on: any audio files ffmpeg reads. A `.json` beside a file may give `{"title": ..., "genre": ...}` for the reports. | Your own files |
| `bench/data/stemgenrt/` | The StemgenRT model files (`drums.onnx`, `bass.onnx`, `vocals.onnx`), for tests and tools that run the real model: set `WLED_DJ_MODELS` to this folder. | `https://github.com/sweetspotsoundsystem/stemgen-rt`. Code MIT; weights CC BY-NC-SA 4.0 (non-commercial, share-alike). Never commit or redistribute them. |
| `bench/data/stems-cache/corpus/` | Separated stems of the files in `bench/corpus/`, written by the tools the first time they need them. | Derived |
| `bench/data/musdb7/`, `bench/data/musdb18/` | MUSDB18 songs with true stems, for `score_detection.py` and `score_layers.py`. | `bench/fetch_musdb.py`, and `musdb18.zip` from Zenodo record 1117372. Academic and non-commercial use. |
| `bench/out/` | What the tools write: reports, timeline pages, look renders. | Derived |
