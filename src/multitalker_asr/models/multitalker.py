import os
from typing import Any, Dict, List, Optional

import torch
from loguru import logger
from nemo.collections.asr.models import ASRModel, EncDecMultiTalkerRNNTBPEModel, SortformerEncLabelModel
from omegaconf import OmegaConf, open_dict

from ..configs import ModelConfig
from ..utils import CheckpointManager, DeviceManager
from .base import BaseASRModel


class MultitalkerASRModel(BaseASRModel):
    def __init__(self, model_cfg: Optional[ModelConfig] = None):
        self._model_cfg = model_cfg or ModelConfig()
        self._device_manager = DeviceManager.from_config(self._model_cfg)
        self._checkpoint_manager = CheckpointManager(self._device_manager.map_location)

        self._asr_model = None
        self._diar_model = None

    @property
    def asr_model(self):
        return self._asr_model

    @property
    def diar_model(self):
        return self._diar_model

    @property
    def device(self) -> torch.device:
        return self._device_manager.device

    @property
    def tokenizer(self):
        return self._asr_model.tokenizer if self._asr_model else None

    @property
    def cfg(self):
        return self._asr_model.cfg if self._asr_model else None

    def load(self, checkpoint_path: Optional[str] = None) -> None:
        self.load_models(checkpoint_path=checkpoint_path)

    def save(self, output_path: str) -> None:
        if self._asr_model is not None:
            self._asr_model.save_to(output_path)
            logger.success(f"Model saved to {output_path}")

    def forward(self, audio: torch.Tensor) -> Dict[str, Any]:
        if self._asr_model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._asr_model(audio)

    def to(self, device: torch.device) -> "MultitalkerASRModel":
        if self._asr_model is not None:
            self._asr_model.to(device)
        if self._diar_model is not None:
            self._diar_model.to(device)
        return self

    def eval(self) -> "MultitalkerASRModel":
        if self._asr_model is not None:
            self._asr_model.eval()
        if self._diar_model is not None:
            self._diar_model.eval()
        return self

    def train(self) -> "MultitalkerASRModel":
        if self._asr_model is not None:
            self._asr_model.train()
        return self

    def load_models(
        self,
        tokenizer_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self._load_diar_model()
        self._load_asr_model(tokenizer_dir, checkpoint_path)

        self._asr_model.to(self._device_manager.device).eval()
        self._diar_model.to(self._device_manager.device).eval()

    def _load_diar_model(self) -> None:
        logger.info(f"Loading Diarization Model from {self._model_cfg.diar_model_path}...")

        if os.path.exists(self._model_cfg.diar_model_path):
            self._diar_model = SortformerEncLabelModel.restore_from(
                restore_path=self._model_cfg.diar_model_path,
                map_location=self._device_manager.map_location,
            )
        else:
            logger.warning(
                f"Diarization model not found at {self._model_cfg.diar_model_path}, trying pretrained..."
            )
            self._diar_model = SortformerEncLabelModel.from_pretrained(
                self._model_cfg.diar_pretrained_name,
                map_location=self._device_manager.map_location,
            )

    def _load_asr_model(
        self,
        tokenizer_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        asr_path = checkpoint_path or self._model_cfg.asr_model_path
        logger.info(f"Loading ASR Model from {asr_path}...")

        if os.path.exists(asr_path):
            if self._checkpoint_manager.is_lightning_checkpoint(asr_path):
                self._load_from_lightning_checkpoint(asr_path, tokenizer_dir)
            else:
                self._asr_model = ASRModel.restore_from(
                    restore_path=asr_path,
                    map_location=self._device_manager.map_location,
                )
        elif self._should_create_from_scratch():
            self._create_from_scratch(tokenizer_dir)
        else:
            logger.warning(f"ASR model not found at {asr_path}, trying cloud...")
            self._asr_model = ASRModel.from_pretrained(
                "nvidia/multitalker-parakeet-streaming-0.6b-v1",
                map_location=self._device_manager.map_location,
            )

    def _load_from_lightning_checkpoint(
        self,
        checkpoint_path: str,
        tokenizer_dir: Optional[str] = None,
    ) -> None:
        logger.info("Detected .ckpt file, loading config and weights manually...")

        cfg, state_dict = self._checkpoint_manager.load_and_extract(
            checkpoint_path, tokenizer_dir
        )

        if cfg is None or state_dict is None:
            raise ValueError("Could not extract config or state_dict from checkpoint.")

        self._asr_model = EncDecMultiTalkerRNNTBPEModel(cfg=cfg)
        self._asr_model.load_state_dict(state_dict)
        self._asr_model.to(self._device_manager.map_location)

    def _should_create_from_scratch(self) -> bool:
        return (
            hasattr(self._model_cfg, "config_path")
            and self._model_cfg.config_path
            and os.path.exists(self._model_cfg.config_path)
        )

    def _create_from_scratch(self, tokenizer_dir: Optional[str] = None) -> None:
        logger.info(
            f"Creating model from scratch using config {self._model_cfg.config_path}..."
        )

        from nemo.collections.common.tokenizers import SentencePieceTokenizer

        cfg = OmegaConf.load(self._model_cfg.config_path)
        vocab_size = getattr(self._model_cfg, "vocab_size", 2048)
        new_vocab_size = vocab_size + 1

        if tokenizer_dir and os.path.exists(tokenizer_dir):
            cfg = self._configure_tokenizer(cfg, tokenizer_dir, vocab_size, new_vocab_size)

        ds_configs = self._remove_dataset_configs(cfg)
        self._asr_model = EncDecMultiTalkerRNNTBPEModel(cfg=cfg, trainer=None)
        self._restore_dataset_configs(ds_configs, tokenizer_dir, new_vocab_size)

        if tokenizer_dir and os.path.exists(tokenizer_dir):
            self._register_tokenizer_artifact(cfg)

        logger.success("Scratch model created successfully!")

    def _configure_tokenizer(
        self,
        cfg,
        tokenizer_dir: str,
        vocab_size: int,
        new_vocab_size: int,
    ):
        from nemo.collections.common.tokenizers import SentencePieceTokenizer

        if os.path.isfile(tokenizer_dir):
            tokenizer_path = tokenizer_dir
        else:
            tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.model")

        tokenizer = SentencePieceTokenizer(model_path=tokenizer_path)
        if tokenizer.vocab_size != vocab_size:
            logger.warning(
                f"Tokenizer vocab size mismatch: {tokenizer.vocab_size} != {vocab_size}"
            )
            vocab_size = tokenizer.vocab_size
            new_vocab_size = vocab_size + 1

        with open_dict(cfg):
            if "tokenizer" not in cfg:
                cfg.tokenizer = {}

            cfg.tokenizer.dir = os.path.dirname(os.path.abspath(tokenizer_path))
            cfg.tokenizer.type = "bpe"
            cfg.tokenizer.model_path = os.path.abspath(tokenizer_path)
            cfg.tokenizer.vocab_size = vocab_size

            vocab_file = os.path.abspath(tokenizer_path).replace(".model", ".vocab")
            if os.path.exists(vocab_file):
                cfg.tokenizer.vocab_path = vocab_file
                cfg.tokenizer.spe_tokenizer_vocab = vocab_file

            if "decoder" in cfg:
                cfg.decoder.vocab_size = new_vocab_size
                if "vocabulary" in cfg.decoder:
                    cfg.decoder.pop("vocabulary")
            if "joint" in cfg:
                cfg.joint.num_classes = new_vocab_size
                if "vocabulary" in cfg.joint:
                    cfg.joint.pop("vocabulary")
            if "model_defaults" in cfg:
                if "vocab_size" in cfg.model_defaults:
                    cfg.model_defaults.vocab_size = vocab_size
                if "num_classes" in cfg.model_defaults:
                    cfg.model_defaults.num_classes = new_vocab_size

        return cfg

    def _remove_dataset_configs(self, cfg) -> Dict[str, Any]:
        ds_configs = {}
        with open_dict(cfg):
            for ds in ["train_ds", "validation_ds", "test_ds"]:
                if ds in cfg:
                    ds_configs[ds] = cfg.pop(ds)
        return ds_configs

    def _restore_dataset_configs(
        self,
        ds_configs: Dict[str, Any],
        tokenizer_dir: Optional[str],
        new_vocab_size: int,
    ) -> None:
        with open_dict(self._asr_model.cfg):
            if tokenizer_dir and os.path.exists(tokenizer_dir):
                if "decoder" in self._asr_model.cfg:
                    self._asr_model.cfg.decoder.vocab_size = new_vocab_size
                if "joint" in self._asr_model.cfg:
                    self._asr_model.cfg.joint.num_classes = new_vocab_size
                if "model_defaults" in self._asr_model.cfg:
                    self._asr_model.cfg.model_defaults.num_classes = new_vocab_size
            for ds, ds_cfg in ds_configs.items():
                self._asr_model.cfg[ds] = ds_cfg

    def _register_tokenizer_artifact(self, cfg) -> None:
        try:
            self._asr_model.register_artifact(
                "tokenizer.model_path", cfg.tokenizer.model_path
            )
        except Exception as e:
            logger.warning(f"Could not register artifact: {e}")

    def change_vocabulary(self, new_tokenizer_dir: str) -> None:
        if os.path.isfile(new_tokenizer_dir):
            new_tokenizer_dir = os.path.dirname(os.path.abspath(new_tokenizer_dir))

        logger.info(f"Changing vocabulary to new tokenizer at {new_tokenizer_dir}...")
        self._asr_model.change_vocabulary(
            new_tokenizer_dir=new_tokenizer_dir, new_tokenizer_type="bpe"
        )

    def set_trainer(self, trainer) -> None:
        self._asr_model.set_trainer(trainer)
