import sys

import vietasr


def main(wav_path: str) -> None:
    pipeline = vietasr.Pipeline.preset("transcribe")
    result = pipeline.transcribe(wav_path)
    print(result.text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sample.wav")
