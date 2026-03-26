from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .base import BaseConfig


class TrainingMode(str, Enum):
    FINETUNE = "finetune"
    TRAIN = "train"
    RESUME = "resume"


@dataclass
class TrainingConfig(BaseConfig):
    mode: str = "finetune"
    train_manifest: str = "data/train.json"
    val_manifest: str = "data/val.json"
    max_steps: int = -1
    max_epochs: int = 50
    learning_rate: float = 1e-5
    weight_decay: float = 1e-3
    batch_size: int = 16
    accumulate_grad_batches: int = 4
    precision: int = 32
    val_check_interval: Optional[int] = None
    output_path: Optional[str] = None
    tokenizer_dir: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    use_on_the_fly_synthesis: bool = False
    max_speakers: int = 4
    synthesis_num_workers: int = 8
    save_every_n_steps: Optional[int] = None
    save_every_n_epochs: Optional[int] = None
    save_top_k: int = 3
    checkpoint_dir: str = "checkpoints"
    log_file_name: str = "training.log"

    @property
    def is_finetune(self) -> bool:
        return self.mode == TrainingMode.FINETUNE

    @property
    def is_train_from_scratch(self) -> bool:
        return self.mode == TrainingMode.TRAIN

    @property
    def is_resume(self) -> bool:
        return self.mode == TrainingMode.RESUME
