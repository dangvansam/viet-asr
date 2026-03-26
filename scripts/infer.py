import argparse

from multitalker_asr.configs import ModelConfig, InferenceConfig
from multitalker_asr.models import MultitalkerASRModel
from multitalker_asr.inference import Transcriber

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multitalker ASR Inference")
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.json")
    parser.add_argument("--model_path", type=str, default="models/multitalker-parakeet-streaming-0.6b-v1.nemo")
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    model_cfg = ModelConfig(
        asr_model_path=args.model_path,
        cuda_id=-1 if args.cpu else 0,
    )

    model = MultitalkerASRModel(model_cfg)
    model.load_models()

    inference_cfg = InferenceConfig(
        audio_file=args.audio,
        output_path=args.output,
    )

    transcriber = Transcriber(model, inference_cfg)
    transcriber.transcribe(args.audio, args.output)
