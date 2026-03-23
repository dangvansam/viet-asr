import argparse
from multitalker_asr.model import MultitalkerASRModel
from multitalker_asr.config import ModelConfig, TrainingConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multitalker ASR Fine-tuning")
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, required=True)
    parser.add_argument("--model_path", type=str,
                        default="models/multitalker-parakeet-streaming-0.6b-v1.nemo")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Max steps. -1 to ignore if using epochs.")
    parser.add_argument("--max_epochs", type=int,
                        default=10, help="Max epochs.")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training. Adjust based on your GPU memory.")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Explicit output path. If None, appends -finetuned.")
    parser.add_argument("--tokenizer_dir", type=str, default=None,
                        help="Path to custom SentencePiece tokenizer directory.")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate. If None, uses model config default.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                        help="Number of batches to accumulate gradients.")
    parser.add_argument("--wandb_project", type=str, default="multitalker-asr",
                        help="WandB project name. Set to empty string to disable.")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="WandB run name.")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to NeMo config YAML (used if creating model from scratch)")
    parser.add_argument("--vocab_size", type=int, default=2048,
                        help="Vocabulary size for from-scratch model creation")

    args = parser.parse_args()

    model_cfg = ModelConfig(
        asr_model_path=args.model_path,
        cuda_id=0 if args.gpus > 0 else -1,
        config_path=args.config_path,
        vocab_size=args.vocab_size
    )

    train_cfg = TrainingConfig(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate if args.learning_rate is not None else 1e-5,
        output_path=args.output_path,
        tokenizer_dir=args.tokenizer_dir,
        accumulate_grad_batches=args.accumulate_grad_batches,
        wandb_project=args.wandb_project if args.wandb_project else None,
        wandb_run_name=args.wandb_run_name
    )

    model = MultitalkerASRModel(model_cfg)
    model.finetune(train_cfg)
