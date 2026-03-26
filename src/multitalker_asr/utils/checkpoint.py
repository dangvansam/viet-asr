import os
from typing import Any, Dict, Optional, Tuple

import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict


class CheckpointManager:
    def __init__(self, map_location: Optional[torch.device] = None):
        self._map_location = map_location or torch.device("cpu")

    def load_checkpoint(
        self,
        checkpoint_path: str,
    ) -> Dict[str, Any]:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self._map_location,
            weights_only=False,
        )
        return checkpoint

    def extract_config_from_checkpoint(
        self,
        checkpoint: Dict[str, Any],
    ) -> Optional[DictConfig]:
        if "hyper_parameters" not in checkpoint:
            return None
        if "cfg" not in checkpoint["hyper_parameters"]:
            return None
        return checkpoint["hyper_parameters"]["cfg"]

    def extract_state_dict(
        self,
        checkpoint: Dict[str, Any],
    ) -> Optional[Dict[str, torch.Tensor]]:
        return checkpoint.get("state_dict")

    def patch_tokenizer_paths(
        self,
        cfg: DictConfig,
        tokenizer_dir: Optional[str] = None,
    ) -> DictConfig:
        if "tokenizer" not in cfg:
            return cfg

        with open_dict(cfg):
            for key in ["model_path", "vocab_path", "spe_tokenizer_vocab"]:
                if key not in cfg.tokenizer:
                    continue
                if not isinstance(cfg.tokenizer[key], str):
                    continue
                if not cfg.tokenizer[key].startswith("nemo:"):
                    continue

                filename = self._extract_nemo_filename(cfg.tokenizer[key])
                base_dir = tokenizer_dir or cfg.tokenizer.get("dir", "")
                local_path = os.path.join(base_dir, filename)

                if os.path.exists(local_path):
                    logger.info(f"Patching {key}: {cfg.tokenizer[key]} -> {local_path}")
                    cfg.tokenizer[key] = local_path

        return cfg

    def _extract_nemo_filename(self, nemo_path: str) -> str:
        if "_" in nemo_path:
            return nemo_path.split("_", 1)[-1]
        return nemo_path[5:]

    def is_lightning_checkpoint(self, path: str) -> bool:
        return path.endswith(".ckpt")

    def is_nemo_checkpoint(self, path: str) -> bool:
        return path.endswith(".nemo")

    def load_and_extract(
        self,
        checkpoint_path: str,
        tokenizer_dir: Optional[str] = None,
    ) -> Tuple[Optional[DictConfig], Optional[Dict[str, torch.Tensor]]]:
        checkpoint = self.load_checkpoint(checkpoint_path)
        cfg = self.extract_config_from_checkpoint(checkpoint)

        if cfg is not None:
            cfg = self.patch_tokenizer_paths(cfg, tokenizer_dir)

        state_dict = self.extract_state_dict(checkpoint)
        return cfg, state_dict
