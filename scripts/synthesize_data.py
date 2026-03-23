import argparse
from multitalker_asr.data.synthesize import synthesize_multitalker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthesize Multi-Speaker Overlapping Data")
    parser.add_argument("--input_manifests", type=str, nargs='+',
                        required=True, help="Single-speaker JSON manifests")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save mixed WAVs")
    parser.add_argument("--output_manifest", type=str,
                        required=True, help="Path for results manifest")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="-1 auto-scales to match input dataset size")
    parser.add_argument("--max_speakers", type=int, default=2)

    args = parser.parse_args()
    synthesize_multitalker(args.input_manifests, args.output_dir, args.output_manifest, args.num_samples, args.max_speakers)
