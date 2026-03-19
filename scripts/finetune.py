import argparse
from multitalker_asr.model import MultitalkerASRModel
from multitalker_asr.config import ModelConfig, TrainingConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multitalker ASR Fine-tuning")
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="models/multitalker-parakeet-streaming-0.6b-v1.nemo")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training. Adjust based on your GPU memory.")

    args = parser.parse_args()

    model_cfg = ModelConfig(
        asr_model_path=args.model_path,
        cuda_id=0 if args.gpus > 0 else -1
    )

    train_cfg = TrainingConfig(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        precision=16 if args.gpus > 0 else 32
    )

    model = MultitalkerASRModel(model_cfg)
    model.finetune(train_cfg)
