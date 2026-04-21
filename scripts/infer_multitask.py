"""CLI entry point for multi-task inference."""
import argparse
import json

from multitalker_asr.configs import ModelConfig, InferenceConfig
from multitalker_asr.configs.multitask import MultiTaskConfig
from multitalker_asr.models.multitask_model import MultitalkerMultiTaskModel
from multitalker_asr.inference.multitask import MultitaskInferenceEngine
from multitalker_asr.inference.post_processor import FunASRPostProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Task Vietnamese ASR Inference")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to audio file")
    parser.add_argument("--model_path", type=str, default="models/multitalker-parakeet-streaming-0.6b-v1.nemo")
    parser.add_argument("--use_post_processor", action="store_true", help="Enable Fun-ASR-MLT-Nano ITN/PnC")
    parser.add_argument("--post_processor_device", type=str, default="cuda:0")
    parser.add_argument("--output_json", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()

    model_cfg = ModelConfig(asr_model_path=args.model_path, device=args.device)
    multitask_cfg = MultiTaskConfig()

    model = MultitalkerMultiTaskModel(model_cfg, multitask_cfg)
    model.load()

    post_processor = None
    if args.use_post_processor:
        post_processor = FunASRPostProcessor(device=args.post_processor_device)

    engine = MultitaskInferenceEngine(
        model=model,
        multitask_cfg=multitask_cfg,
        post_processor=post_processor,
    )

    results = engine.infer(args.audio_path)

    for r in results:
        print(r)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
