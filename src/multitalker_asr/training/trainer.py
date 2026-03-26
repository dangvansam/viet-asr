import os
from typing import List, Optional, Union

import lightning.pytorch as pl
import lightning.pytorch.trainer.trainer as pl_trainer
import torch
from loguru import logger
from omegaconf import OmegaConf, open_dict

from ..configs import ModelConfig, TrainingConfig
from ..configs.training import TrainingMode
from ..data import DataLoaderFactory
from ..models import MultitalkerASRModel
from .base import BaseTrainer
from .callbacks import PrintLossCallback


class MultitalkerTrainer(BaseTrainer):
    def __init__(
        self,
        model: MultitalkerASRModel,
        train_cfg: TrainingConfig,
        model_cfg: Optional[ModelConfig] = None,
    ):
        self._model = model
        self._train_cfg = train_cfg
        self._model_cfg = model_cfg or ModelConfig()
        self._trainer = None
        self._train_loader = None
        self._val_loader = None

    @property
    def mode(self) -> str:
        return self._train_cfg.mode

    def setup(self) -> None:
        if not hasattr(self, "_run_name_configured"):
            if self.mode != TrainingMode.RESUME:
                run_name = self._train_cfg.wandb_run_name
                if not run_name:
                    from datetime import datetime
                    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                self._train_cfg.checkpoint_dir = os.path.join(
                    self._train_cfg.checkpoint_dir, run_name
                )
                if self._train_cfg.wandb_run_name is None:
                    self._train_cfg.wandb_run_name = run_name
            self._run_name_configured = True

        self._configure_logging()
        self._save_config()

        if self._model.asr_model is None:
            tokenizer_dir = getattr(self._train_cfg, "tokenizer_dir", None)
            self._model.load_models(tokenizer_dir=tokenizer_dir)

        self._maybe_change_vocabulary()
        self._configure_datasets()
        self._setup_data_loaders()
        self._configure_optimization()
        self._disable_cuda_graphs()
        self._create_trainer()

        logger.info(f"Training mode: {self.mode}")

    def train(self) -> None:
        if self._trainer is None:
            self.setup()

        self._model.asr_model.set_trainer(self._trainer)
        pl_trainer._maybe_unwrap_optimized = lambda x: x
        self._model.asr_model.train()

        ckpt_path = self._get_resume_checkpoint_path()

        if self._train_cfg.use_on_the_fly_synthesis:
            self._trainer.fit(
                self._model.asr_model,
                train_dataloaders=self._train_loader,
                val_dataloaders=self._val_loader,
                ckpt_path=ckpt_path,
            )
        else:
            self._trainer.fit(self._model.asr_model, ckpt_path=ckpt_path)

    def save(self, output_path: Optional[str] = None) -> str:
        save_path = self._resolve_output_path(output_path)
        self._model.save(save_path)
        return save_path

    def train_and_save(self, output_path: Optional[str] = None) -> str:
        self.setup()
        self.train()
        return self.save(output_path)

    def _resolve_output_path(self, output_path: Optional[str] = None) -> str:
        if output_path:
            return output_path
        if self._train_cfg.output_path:
            return self._train_cfg.output_path

        suffix_map = {
            TrainingMode.FINETUNE: "-finetuned.nemo",
            TrainingMode.TRAIN: "-trained.nemo",
            TrainingMode.RESUME: "-resumed.nemo",
        }
        suffix = suffix_map.get(self._train_cfg.mode, "-output.nemo")

        base = self._model_cfg.asr_model_path
        for ext in (".nemo", ".ckpt"):
            if base.endswith(ext):
                return base.replace(ext, suffix)
        return base + suffix

    def _maybe_change_vocabulary(self) -> None:
        tokenizer_dir = getattr(self._train_cfg, "tokenizer_dir", None)
        if tokenizer_dir is None:
            return

        if self._train_cfg.is_train_from_scratch or self._train_cfg.is_resume:
            return

        is_from_scratch = (
            hasattr(self._model_cfg, "config_path")
            and self._model_cfg.config_path
            and not os.path.exists(self._model_cfg.asr_model_path)
        )
        is_resuming = (
            self._model_cfg.asr_model_path.endswith(".ckpt")
            if self._model_cfg.asr_model_path
            else False
        )

        if is_from_scratch or is_resuming:
            return

        self._model.change_vocabulary(tokenizer_dir)

    def _configure_datasets(self) -> None:
        cfg = self._model.asr_model.cfg
        with open_dict(cfg):
            cfg.train_ds.batch_size = self._train_cfg.batch_size
            cfg.train_ds.text_field = "text"

            self._configure_manifest(cfg.train_ds, self._train_cfg.train_manifest)

            if not hasattr(cfg, "validate_ds") or cfg.validate_ds is None:
                cfg.validate_ds = cfg.train_ds.copy()

            cfg.validate_ds.batch_size = self._train_cfg.batch_size
            cfg.validate_ds.text_field = "text"

            self._configure_manifest(cfg.validate_ds, self._train_cfg.val_manifest)

            if self._train_cfg.learning_rate and self._train_cfg.learning_rate != 1e-5:
                cfg.optim.lr = self._train_cfg.learning_rate

            if self._train_cfg.weight_decay and self._train_cfg.weight_decay != 1e-3:
                cfg.optim.weight_decay = self._train_cfg.weight_decay

    def _configure_manifest(self, ds_cfg, manifest_path: str) -> None:
        is_lhotse = manifest_path.endswith(".jsonl") or manifest_path.endswith(".jsonl.gz")

        if is_lhotse:
            ds_cfg.use_lhotse = True
            ds_cfg.cuts_path = manifest_path
            ds_cfg.manifest_filepath = None
        else:
            ds_cfg.use_lhotse = False
            ds_cfg.cuts_path = None
            ds_cfg.manifest_filepath = manifest_path

    def _setup_data_loaders(self) -> None:
        if not self._train_cfg.use_on_the_fly_synthesis:
            cfg = self._model.asr_model.cfg
            self._model.asr_model.setup_training_data(train_data_config=cfg.train_ds)
            self._model.asr_model.setup_multiple_validation_data(val_data_config=cfg.validate_ds)
            return

        logger.info("Setting up on-the-fly synthesis data loaders...")

        train_manifests = self._normalize_manifests(self._train_cfg.train_manifest)
        val_manifests = self._normalize_manifests(self._train_cfg.val_manifest)

        tokenizer = self._model.tokenizer

        self._train_loader = DataLoaderFactory.create_streaming_dataloader(
            manifest_paths=train_manifests,
            tokenizer=tokenizer,
            batch_size=self._train_cfg.batch_size,
            max_speakers=self._train_cfg.max_speakers,
            num_workers=self._train_cfg.synthesis_num_workers,
        )

        self._val_loader = DataLoaderFactory.create_streaming_dataloader(
            manifest_paths=val_manifests,
            tokenizer=tokenizer,
            batch_size=self._train_cfg.batch_size,
            max_speakers=self._train_cfg.max_speakers,
            num_workers=self._train_cfg.synthesis_num_workers,
            max_samples=self._train_cfg.batch_size * 20,
        )

        self._model.asr_model._train_dl = self._train_loader
        self._model.asr_model._validation_dl = (
            self._val_loader
            if isinstance(self._val_loader, list)
            else [self._val_loader]
        )

    def _configure_optimization(self) -> None:
        cfg = self._model.asr_model.cfg
        self._model.asr_model.setup_optimization(optim_config=cfg.optim)

    def _disable_cuda_graphs(self) -> None:
        if hasattr(self._model.asr_model, "cfg"):
            with open_dict(self._model.asr_model.cfg):
                self._model.asr_model.cfg.enable_cuda_graphs = False
                
    def _configure_logging(self) -> None:
        os.makedirs(self._train_cfg.checkpoint_dir, exist_ok=True)
        log_file = os.path.join(
            self._train_cfg.checkpoint_dir, self._train_cfg.log_file_name
        )
        logger.add(log_file, rotation="10 MB", level="INFO")
        logger.info(f"Logging to {log_file}")

    def _save_config(self) -> None:
        os.makedirs(self._train_cfg.checkpoint_dir, exist_ok=True)
        
        # Save ModelConfig
        model_cfg_path = os.path.join(self._train_cfg.checkpoint_dir, "model_config.yaml")
        if self._model_cfg:
            import dataclasses
            with open(model_cfg_path, "w") as f:
                OmegaConf.save(OmegaConf.create(dataclasses.asdict(self._model_cfg)), f)
        
        # Save TrainingConfig
        train_cfg_path = os.path.join(self._train_cfg.checkpoint_dir, "train_config.yaml")
        import dataclasses
        with open(train_cfg_path, "w") as f:
            OmegaConf.save(OmegaConf.create(dataclasses.asdict(self._train_cfg)), f)
            
        # Save NeMo Model Config (if available)
        if hasattr(self._model.asr_model, "cfg"):
            nemo_cfg_path = os.path.join(self._train_cfg.checkpoint_dir, "nemo_model_config.yaml")
            OmegaConf.save(self._model.asr_model.cfg, nemo_cfg_path)

    def _create_trainer(self) -> None:
        callbacks = [PrintLossCallback()]
        callbacks.append(self._create_checkpoint_callback())
        loggers = self._create_loggers()

        max_epochs = self._train_cfg.max_epochs
        if self._train_cfg.max_steps > 0:
            max_epochs = -1

        self._trainer = pl.Trainer(
            devices=1,
            accelerator=self._get_accelerator(),
            max_steps=self._train_cfg.max_steps,
            max_epochs=max_epochs,
            accumulate_grad_batches=self._train_cfg.accumulate_grad_batches,
            precision=self._train_cfg.precision,
            val_check_interval=self._train_cfg.val_check_interval,
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            default_root_dir=self._train_cfg.checkpoint_dir,
            log_every_n_steps=20,
            inference_mode=False,
            callbacks=callbacks,
            logger=loggers,
            limit_val_batches=20,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
        )

    def _create_checkpoint_callback(self) -> pl.callbacks.ModelCheckpoint:
        kwargs = {
            "dirpath": self._train_cfg.checkpoint_dir,
            "save_top_k": self._train_cfg.save_top_k,
            "monitor": "val_loss",
            "mode": "min",
            "save_last": True,
        }

        if self._train_cfg.save_every_n_steps:
            kwargs["every_n_train_steps"] = self._train_cfg.save_every_n_steps
        elif self._train_cfg.save_every_n_epochs:
            kwargs["every_n_epochs"] = self._train_cfg.save_every_n_epochs

        return pl.callbacks.ModelCheckpoint(**kwargs)

    def _create_loggers(self):
        if hasattr(self._train_cfg, "wandb_project") and self._train_cfg.wandb_project:
            from lightning.pytorch.loggers import WandbLogger

            return [
                WandbLogger(
                    project=self._train_cfg.wandb_project,
                    name=self._train_cfg.wandb_run_name,
                    save_dir=self._train_cfg.checkpoint_dir,
                )
            ]
        return True

    def _get_accelerator(self) -> str:
        if torch.cuda.is_available() and self._model_cfg.cuda_id >= 0:
            return "gpu"
        return "cpu"

    def _get_resume_checkpoint_path(self) -> Optional[str]:
        if self._train_cfg.is_resume:
            return self._model_cfg.asr_model_path

        if (
            self._model_cfg.asr_model_path
            and self._model_cfg.asr_model_path.endswith(".ckpt")
        ):
            return self._model_cfg.asr_model_path
        return None

    def _normalize_manifests(
        self, manifest: Union[str, List[str]]
    ) -> List[str]:
        if isinstance(manifest, str):
            return [manifest]
        return manifest
