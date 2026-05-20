import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import vietasr


def stream_file(wav_path: str, chunk_ms: int = 320) -> None:
    audio, sr = sf.read(wav_path, dtype="int16")
    if audio.ndim == 2:
        audio = audio.mean(axis=1).astype("int16")
    duration_s = len(audio) / sr
    print(f"audio: {duration_s:.2f}s @ {sr} Hz, {len(audio)} samples")
    print()

    pipeline = vietasr.Pipeline.preset("transcribe")

    chunk_samples = int(sr * chunk_ms / 1000)
    started = time.perf_counter()
    last_partial = ""

    with pipeline.stream(sample_rate=float(sr)) as session:
        for offset in range(0, len(audio), chunk_samples):
            chunk = audio[offset:offset + chunk_samples]
            session.accept(np.asarray(chunk, dtype=np.int16))
            partial = session.partial().text
            if partial != last_partial:
                t = (offset + len(chunk)) / sr
                print(f"  [{t:5.2f}s] partial: {partial}")
                last_partial = partial
        final = session.final().text
        elapsed = time.perf_counter() - started

    print()
    print(f"FINAL ({elapsed:.2f}s wall, RTF {elapsed/duration_s:.2f}):")
    print(final)


if __name__ == "__main__":
    stream_file(sys.argv[1] if len(sys.argv) > 1 else "audio.wav")
