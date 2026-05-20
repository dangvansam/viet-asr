import argparse
import json
import sys

from vietasr import Pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="vietasr",
        description="Universal Vietnamese Speech AI SDK CLI",
    )
    parser.add_argument("wav_path", help="Path to a 16-bit PCM WAV file")
    parser.add_argument("--preset", default="transcribe",
                        help="Preset name (transcribe | transcribe-rich | meeting | analytics)")
    parser.add_argument("--module", action="append", default=[],
                        help="Add module to a custom pipeline (overrides --preset)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()

    if args.module:
        pipeline = Pipeline.new()
        for module_name in args.module:
            pipeline.add(module_name)
        pipeline.build()
    else:
        pipeline = Pipeline.preset(args.preset)

    result = pipeline.transcribe(args.wav_path)
    if args.pretty:
        print(json.dumps(result.payload, ensure_ascii=False, indent=2))
    else:
        print(result.to_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
