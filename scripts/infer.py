import argparse
from multitalker_asr.model import MultitalkerASRModel
from multitalker_asr.config import ModelConfig, InferenceConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multitalker ASR Inference")
    parser.add_argument("--audio", type=str, required=True,
                        help="Path to audio file")
    parser.add_argument("--output", type=str,
                        default="output.json", help="Output JSON path")
    parser.add_argument("--model_path", type=str,
                        default="models/multitalker-parakeet-streaming-0.6b-v1.nemo")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")

    args = parser.parse_args()

    model_cfg = ModelConfig(
        asr_model_path=args.model_path,
        cuda_id=-1 if args.cpu else 0
    )

    model = MultitalkerASRModel(model_cfg)
    model.load_models()

    infer_cfg = InferenceConfig(
        audio_file=args.audio,
        output_path=args.output
    )

    model.transcribe(args.audio, args.output, infer_cfg)
