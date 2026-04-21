"""CLI entry point for multi-task curriculum training."""
import argparse

from multitalker_asr.configs import ModelConfig, TrainingConfig
from multitalker_asr.configs.multitask import MultiTaskConfig
from multitalker_asr.models.multitask_model import MultitalkerMultiTaskModel
from multitalker_asr.training.curriculum_trainer import CurriculumTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Task Vietnamese ASR Training")

    # Model args
    parser.add_argument("--model_path", type=str, default="models/multitalker-parakeet-streaming-0.6b-v1.nemo")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--tokenizer_dir", type=str, default=None)
    parser.add_argument("--vocab_size", type=int, default=2048)
    parser.add_argument("--encoder_source", type=str, default="nemo", choices=["nemo", "funasr", "scratch"])

    # Training args
    parser.add_argument("--mode", type=str, default="finetune", choices=["finetune", "train", "resume"])
    parser.add_argument("--curriculum_phase", type=str, default="asr", choices=["asr", "multitalker", "paralinguistic"])
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, required=True)
    parser.add_argument("--multitask_train_manifest", type=str, default=None, help="Annotated manifest for Phase 3")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--max_speakers", type=int, default=2)

    # Curriculum args
    parser.add_argument("--num_frozen_layers", type=int, default=18)
    parser.add_argument("--use_dynamic_weighting", action="store_true", default=True)
    parser.add_argument("--prompt_ce_weight", type=float, default=1.0)

    # Multi-task args
    parser.add_argument("--prompt_embed_dim", type=int, default=80)
    parser.add_argument("--emotion_classes", type=int, default=7)
    parser.add_argument("--gender_classes", type=int, default=2)
    parser.add_argument("--age_classes", type=int, default=4)
    parser.add_argument("--voice_state_classes", type=int, default=2)

    # Output args
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="multitalker-asr")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    model_cfg = ModelConfig(
        asr_model_path=args.model_path,
        config_path=args.config_path,
        vocab_size=args.vocab_size,
    )

    multitask_cfg = MultiTaskConfig(
        prompt_embed_dim=args.prompt_embed_dim,
        emotion_classes=args.emotion_classes,
        gender_classes=args.gender_classes,
        age_classes=args.age_classes,
        voice_state_classes=args.voice_state_classes,
        ce_loss_weight=args.prompt_ce_weight,
        encoder_source=args.encoder_source,
    )

    train_cfg = TrainingConfig(
        mode=args.mode,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_speakers=args.max_speakers,
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output_path,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        tokenizer_dir=args.tokenizer_dir,
    )

    # Attach curriculum-specific attrs
    train_cfg.curriculum_phase = args.curriculum_phase
    train_cfg.num_frozen_layers = args.num_frozen_layers
    train_cfg.use_dynamic_weighting = args.use_dynamic_weighting
    train_cfg.multitask_train_manifest = args.multitask_train_manifest

    if args.learning_rate is not None:
        train_cfg.learning_rate = args.learning_rate

    model = MultitalkerMultiTaskModel(model_cfg, multitask_cfg)
    trainer = CurriculumTrainer(model, train_cfg, model_cfg, multitask_cfg)
    trainer.setup()
    trainer.train()

    if args.output_path:
        model.save(args.output_path)


if __name__ == "__main__":
    main()
