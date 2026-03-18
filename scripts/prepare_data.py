import argparse
from multitalker_asr.data.prepare import create_nemo_manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create NeMo manifest from transcript CSV")
    parser.add_argument("--csv", type=str, required=True,
                        help="Input CSV file")
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory containing audio files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON manifest path")

    args = parser.parse_args()
    create_nemo_manifest(args.csv, args.output, args.audio_dir)
