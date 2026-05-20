import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import soundfile as sf

import vietasr


def stream_one(pipe, audio, sr, chunk_ms, worker_id):
    chunk_samples = int(sr * chunk_ms / 1000)
    with pipe.stream(sample_rate=float(sr)) as session:
        for offset in range(0, len(audio), chunk_samples):
            chunk = audio[offset:offset + chunk_samples]
            session.accept(np.asarray(chunk, dtype=np.int16))
        return worker_id, session.final().text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-ms", type=int, default=320)
    args = parser.parse_args()

    audio, sr = sf.read(args.wav_path, dtype="int16")
    if audio.ndim == 2:
        audio = audio.mean(axis=1).astype("int16")

    pipe = vietasr.Pipeline.preset("transcribe")

    print(f"{args.workers} concurrent Sessions on ONE shared Pipeline, "
          f"{len(audio)/sr:.1f}s audio each, {args.chunk_ms} ms chunks")

    results = {}
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(stream_one, pipe, audio, sr, args.chunk_ms, i)
            for i in range(args.workers)
        ]
        for fut in as_completed(futures):
            try:
                wid, text = fut.result()
                results[wid] = text
            except Exception as exc:
                errors.append(repr(exc))

    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)

    unique = set(results.values())
    print(f"unique transcripts: {len(unique)}")
    for wid, text in sorted(results.items()):
        print(f"  worker {wid}: {text[:80]}...")

    if len(unique) == 1:
        print("OK — all workers produced identical, correct transcripts")
    else:
        print("FAIL — different transcripts:")
        for t in unique:
            print(f"    {t!r}")
        sys.exit(2)


if __name__ == "__main__":
    main()
