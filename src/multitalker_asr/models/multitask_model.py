from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from loguru import logger

from ..configs import ModelConfig
from ..configs.conditioning import (
    ConditionerFactory,
    ConditioningConfig,
    ConditioningStrategy,
)
from ..configs.multitask import MultiTaskConfig
from ..training.losses import MultiTaskLoss
from .base import BaseASRModel
from .conditioning import (
    AuxLossScheduler,
    BaseConditioner,
    PrependPromptCEConditioner,
)
from .multitalker import MultitalkerASRModel
from .prompt_embedding import PromptEmbedding, TaskTokenRegistry


class MultitalkerMultiTaskModel(BaseASRModel):
    """Wraps MultitalkerASRModel with pluggable attribute conditioning.

    Conditioning strategy is selected via ConditioningConfig:
        * PREPEND_PROMPT_CE — SenseVoice-style prepended prompt tokens + CE head (default).
        * DECODER_TAG_ONLY  — no encoder change, tags in target vocab.
        * FEATURE_CONCAT    — Nemotron-style broadcast + concat along feature dim.
        * HYBRID            — feature_concat + decoder tags + decayed aux head.
    """

    def __init__(
        self,
        model_cfg: Optional[ModelConfig] = None,
        multitask_cfg: Optional[MultiTaskConfig] = None,
        conditioning_cfg: Optional[ConditioningConfig] = None,
    ):
        self._model_cfg = model_cfg or ModelConfig()
        self._multitask_cfg = multitask_cfg or MultiTaskConfig()
        self._conditioning_cfg = conditioning_cfg or ConditioningConfig(
            strategy=ConditioningStrategy.PREPEND_PROMPT_CE
        )

        self._base_model = MultitalkerASRModel(self._model_cfg)
        self._registry = TaskTokenRegistry(
            self._multitask_cfg,
            prefer_legacy_labels=(
                self._conditioning_cfg.strategy
                == ConditioningStrategy.PREPEND_PROMPT_CE
            ),
        )
        self._conditioner: BaseConditioner = ConditionerFactory.build(
            self._conditioning_cfg,
            multitask_config=self._multitask_cfg,
            registry=self._registry,
        )
        self._aux_scheduler = AuxLossScheduler(
            initial_weight=self._conditioning_cfg.aux_head_loss_weight,
            decay_epochs=self._conditioning_cfg.aux_head_loss_decay_epochs,
            min_weight=0.0,
        )
        self._current_epoch = 0
        self._multi_task_loss = MultiTaskLoss(
            rnnt_weight=1.0,
            prompt_weight=self._multitask_cfg.ce_loss_weight,
        )

    @property
    def base_model(self) -> MultitalkerASRModel:
        return self._base_model

    @property
    def asr_model(self):
        return self._base_model.asr_model

    @property
    def diar_model(self):
        return self._base_model.diar_model

    @property
    def conditioner(self) -> BaseConditioner:
        return self._conditioner

    @property
    def conditioning_config(self) -> ConditioningConfig:
        return self._conditioning_cfg

    @property
    def prompt_embedding(self) -> PromptEmbedding:
        if isinstance(self._conditioner, PrependPromptCEConditioner):
            return self._conditioner._prompt_embed
        raise AttributeError(
            "prompt_embedding is only available for PREPEND_PROMPT_CE strategy"
        )

    @property
    def registry(self) -> TaskTokenRegistry:
        return self._registry

    @property
    def device(self) -> torch.device:
        return self._base_model.device

    @property
    def tokenizer(self):
        return self._base_model.tokenizer

    @property
    def cfg(self):
        return self._base_model.cfg

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = max(0, int(epoch))

    def aux_loss_weight(self) -> float:
        return self._aux_scheduler.weight(self._current_epoch)

    def load(self, checkpoint_path: Optional[str] = None) -> None:
        self.load_models(checkpoint_path=checkpoint_path)

    def save(self, output_path: str) -> None:
        self._base_model.save(output_path)

    def forward(self, audio: torch.Tensor) -> Dict[str, Any]:
        return self._base_model.forward(audio)

    def to(self, device: torch.device) -> "MultitalkerMultiTaskModel":
        self._base_model.to(device)
        self._conditioner.to(device)
        return self

    def eval(self) -> "MultitalkerMultiTaskModel":
        self._base_model.eval()
        self._conditioner.eval()
        return self

    def train(self) -> "MultitalkerMultiTaskModel":
        self._base_model.train()
        self._conditioner.train()
        return self

    def load_models(
        self,
        tokenizer_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self._base_model.load_models(tokenizer_dir, checkpoint_path)
        self._init_conditioner_layers()

    def _init_conditioner_layers(self) -> None:
        encoder_hidden = self._get_encoder_hidden_dim()
        encoder_input = self._get_encoder_input_dim()
        self._conditioner.initialize_layers(encoder_input, encoder_hidden)
        logger.info(
            f"Conditioner '{self._conditioner.name}' initialized "
            f"(input_dim={encoder_input}, hidden_dim={encoder_hidden})"
        )

    def _get_encoder_hidden_dim(self) -> int:
        asr = self._base_model.asr_model
        if hasattr(asr, "encoder") and hasattr(asr.encoder, "_feat_out"):
            return asr.encoder._feat_out
        if hasattr(asr, "cfg") and "encoder" in asr.cfg:
            enc_cfg = asr.cfg.encoder
            if hasattr(enc_cfg, "d_model"):
                return enc_cfg.d_model
        if hasattr(asr, "cfg") and "model_defaults" in asr.cfg:
            if hasattr(asr.cfg.model_defaults, "enc_hidden"):
                return asr.cfg.model_defaults.enc_hidden
        raise RuntimeError("Cannot determine encoder hidden dim from loaded model")

    def _get_encoder_input_dim(self) -> int:
        asr = self._base_model.asr_model
        if hasattr(asr, "cfg") and "preprocessor" in asr.cfg:
            if hasattr(asr.cfg.preprocessor, "features"):
                return asr.cfg.preprocessor.features
        return self._multitask_cfg.prompt_embed_dim

    def predict_prompt_labels(
        self, encoder_out: torch.Tensor, state=None
    ) -> Optional[Dict[str, torch.Tensor]]:
        if state is None:
            from .conditioning.base import ConditionState

            num_pos = getattr(self._conditioner, "num_positions", 0)
            state = ConditionState(num_prepended=num_pos)
        return self._conditioner.predict_attributes(encoder_out, state)

    def forward_multitask(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        task_labels: Optional[Dict[str, torch.Tensor]] = None,
        text: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        asr = self._base_model.asr_model

        processed, processed_lens = asr.preprocessor(
            input_signal=audio, length=audio_lengths
        )

        if (
            asr.training
            and hasattr(asr, "spec_augmentation")
            and asr.spec_augmentation is not None
        ):
            processed = asr.spec_augmentation(
                input_spec=processed, length=processed_lens
            )

        augmented, augmented_lens, state = self._conditioner.apply_input(
            processed, processed_lens, task_labels
        )

        encoded, encoded_lens = asr.encoder(
            audio_signal=augmented, length=augmented_lens
        )

        result: Dict[str, Any] = {}

        if task_labels is not None:
            loss_prompt, accuracy = self._conditioner.compute_aux_loss(
                encoded, task_labels, state
            )
            loss_prompt = loss_prompt * self.aux_loss_weight()
            result["loss_prompt"] = loss_prompt
            result["accuracy"] = accuracy
        else:
            loss_prompt = torch.zeros((), device=audio.device)
            result["loss_prompt"] = loss_prompt
            result["accuracy"] = {}

        preds = self._conditioner.predict_attributes(encoded, state)
        if preds is not None:
            result["prompt_preds"] = preds

        if text is not None and text_lengths is not None:
            speech_encoded, speech_encoded_lens = self._conditioner.strip_speech(
                encoded, encoded_lens, state
            )

            decoder_out, _, _ = asr.decoder(
                targets=text, target_length=text_lengths
            )
            joint_out = asr.joint(
                encoder_outputs=speech_encoded,
                decoder_outputs=decoder_out,
            )
            loss_rnnt = asr.loss(
                log_probs=joint_out,
                targets=text,
                input_lengths=speech_encoded_lens,
                target_lengths=text_lengths,
            )
            result["loss_rnnt"] = loss_rnnt
        else:
            loss_rnnt = torch.zeros((), device=audio.device)
            result["loss_rnnt"] = loss_rnnt

        total_loss, loss_stats = self._multi_task_loss(loss_rnnt, loss_prompt)
        result["loss_total"] = total_loss
        result["loss_stats"] = loss_stats
        result["condition_state"] = state

        return result

    def freeze_encoder_layers(self, num_layers: int) -> None:
        asr = self._base_model.asr_model
        if not hasattr(asr, "encoder") or not hasattr(asr.encoder, "layers"):
            logger.warning("Encoder does not expose layers — cannot freeze selectively")
            return
        total = len(asr.encoder.layers)
        n = min(num_layers, total)
        for i, layer in enumerate(asr.encoder.layers[:n]):
            for param in layer.parameters():
                param.requires_grad = False
        logger.info(f"Frozen {n}/{total} encoder layers")

    def unfreeze_all(self) -> None:
        for param in self._base_model.asr_model.parameters():
            param.requires_grad = True
        for param in self._conditioner.parameters():
            param.requires_grad = True
        logger.info("All parameters unfrozen")

    def set_trainer(self, trainer) -> None:
        self._base_model.set_trainer(trainer)

    def change_vocabulary(self, new_tokenizer_dir: str) -> None:
        self._base_model.change_vocabulary(new_tokenizer_dir)

    @staticmethod
    def load_encoder_from_funasr(
        funasr_checkpoint_path: str,
        nemo_model: "MultitalkerMultiTaskModel",
        strict: bool = False,
    ) -> None:
        state_dict = torch.load(funasr_checkpoint_path, map_location="cpu")
        encoder_state = {}
        for key, value in state_dict.items():
            if key.startswith("audio_encoder."):
                new_key = key.replace("audio_encoder.", "")
                encoder_state[new_key] = value

        if not encoder_state:
            raise RuntimeError(
                f"No 'audio_encoder.*' keys found in {funasr_checkpoint_path}"
            )

        result = nemo_model.asr_model.encoder.load_state_dict(
            encoder_state, strict=strict
        )
        if result.missing_keys:
            logger.warning(f"Missing keys in encoder: {result.missing_keys[:10]}...")
        if result.unexpected_keys:
            logger.warning(
                f"Unexpected keys from FunASR: {result.unexpected_keys[:10]}..."
            )
        logger.success(
            f"Loaded {len(encoder_state)} encoder params from FunASR checkpoint"
        )
