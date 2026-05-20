import sys

import numpy as np
import soundfile as sf

import vietasr


def main(wav_path: str) -> None:
    pipeline = vietasr.Pipeline.preset("transcribe")
    audio, sr = sf.read(wav_path, dtype="int16")
    chunk_size = int(sr * 0.5)

    with pipeline.stream(sample_rate=float(sr)) as session:
        for offset in range(0, len(audio), chunk_size):
            chunk = audio[offset:offset + chunk_size]
            session.accept(np.asarray(chunk, dtype=np.int16))
            print(session.partial().text, flush=True)
        print("FINAL:", session.final().text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sample.wav")
