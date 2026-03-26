import argparse

from multitalker_asr.data import MultitalkerSynthesizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesize Multi-Speaker Overlapping Data")
    parser.add_argument("--input_manifests", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_manifest", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--max_speakers", type=int, default=2)

    args = parser.parse_args()

    synthesizer = MultitalkerSynthesizer()
    synthesizer.synthesize(
        input_manifests=args.input_manifests,
        output_dir=args.output_dir,
        output_manifest=args.output_manifest,
        num_samples=args.num_samples,
        max_speakers=args.max_speakers,
    )
