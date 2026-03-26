import argparse

from multitalker_asr.configs import ModelConfig, TrainingConfig
from multitalker_asr.configs.training import TrainingMode
from multitalker_asr.models import MultitalkerASRModel
from multitalker_asr.training import MultitalkerTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Multitalker ASR Training")

    parser.add_argument(
        "--mode",
        type=str,
        default="finetune",
        choices=["finetune", "train", "resume"],
    )

    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, required=True)
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/multitalker-parakeet-streaming-0.6b-v1.nemo",
    )
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--tokenizer_dir", type=str, default=None)
    parser.add_argument("--vocab_size", type=int, default=2048)

    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=1)

    parser.add_argument("--use_on_the_fly_synthesis", action="store_true")
    parser.add_argument("--max_speakers", type=int, default=2)
    parser.add_argument("--synthesis_num_workers", type=int, default=4)

    parser.add_argument("--save_every_n_steps", type=int, default=None)
    parser.add_argument("--save_every_n_epochs", type=int, default=None)
    parser.add_argument("--save_top_k", type=int, default=3)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_file_name", type=str, default="training.log")

    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="multitalker-asr")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_cfg = ModelConfig(
        asr_model_path=args.model_path,
        cuda_id=0 if args.gpus > 0 else -1,
        config_path=args.config_path,
        vocab_size=args.vocab_size,
    )

    train_cfg = TrainingConfig(
        mode=args.mode,
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
        wandb_run_name=args.wandb_run_name,
        use_on_the_fly_synthesis=args.use_on_the_fly_synthesis,
        max_speakers=args.max_speakers,
        synthesis_num_workers=args.synthesis_num_workers,
        save_every_n_steps=args.save_every_n_steps,
        save_every_n_epochs=args.save_every_n_epochs,
        save_top_k=args.save_top_k,
        checkpoint_dir=args.checkpoint_dir,
        log_file_name=args.log_file_name,
    )

    model = MultitalkerASRModel(model_cfg)
    trainer = MultitalkerTrainer(model, train_cfg, model_cfg)
    trainer.train_and_save()
