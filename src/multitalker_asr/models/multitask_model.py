from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger

from ..configs import ModelConfig
from ..configs.multitask import MultiTaskConfig
from ..training.losses import MultiTaskLoss
from .base import BaseASRModel
from .multitalker import MultitalkerASRModel
from .prompt_embedding import PromptEmbedding, TaskTokenRegistry


class MultitalkerMultiTaskModel(BaseASRModel):
    """Wraps MultitalkerASRModel with SenseVoice-style prompt-based multi-task capability.

    Architecture:
        [prompt_tokens: lang, emotion, gender, age, voice_state, textnorm] + [speech]
          → Encoder → encoder_out
            ├── [:, :6, :] → CE loss on prompt positions (paralinguistic classification)
            └── [:, 6:, :] → RNNT loss (ASR)

    Reference: funasr/models/sense_voice/model.py:642-807
    """

    def __init__(
        self,
        model_cfg: Optional[ModelConfig] = None,
        multitask_cfg: Optional[MultiTaskConfig] = None,
    ):
        self._model_cfg = model_cfg or ModelConfig()
        self._multitask_cfg = multitask_cfg or MultiTaskConfig()

        self._base_model = MultitalkerASRModel(self._model_cfg)
        self._prompt_embed = PromptEmbedding(self._multitask_cfg)
        self._registry = self._prompt_embed.registry

        # Initialized after model loading (need encoder hidden dim)
        self._prompt_classifier: Optional[nn.Linear] = None
        self._prompt_loss_fn = nn.CrossEntropyLoss()
        self._multi_task_loss = MultiTaskLoss(
            rnnt_weight=1.0,
            prompt_weight=self._multitask_cfg.ce_loss_weight,
        )

        # Projection layer if prompt embed dim != encoder input dim
        self._prompt_projection: Optional[nn.Linear] = None

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
    def prompt_embedding(self) -> PromptEmbedding:
        return self._prompt_embed

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

    def load(self, checkpoint_path: Optional[str] = None) -> None:
        self.load_models(checkpoint_path=checkpoint_path)

    def save(self, output_path: str) -> None:
        self._base_model.save(output_path)

    def forward(self, audio: torch.Tensor) -> Dict[str, Any]:
        return self._base_model.forward(audio)

    def to(self, device: torch.device) -> "MultitalkerMultiTaskModel":
        self._base_model.to(device)
        self._prompt_embed.to(device)
        if self._prompt_classifier is not None:
            self._prompt_classifier.to(device)
        if self._prompt_projection is not None:
            self._prompt_projection.to(device)
        return self

    def eval(self) -> "MultitalkerMultiTaskModel":
        self._base_model.eval()
        self._prompt_embed.eval()
        return self

    def train(self) -> "MultitalkerMultiTaskModel":
        self._base_model.train()
        self._prompt_embed.train()
        return self

    def load_models(
        self,
        tokenizer_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """Load base ASR + diarization models, then initialize prompt layers."""
        self._base_model.load_models(tokenizer_dir, checkpoint_path)
        self._init_prompt_layers()

    def _init_prompt_layers(self) -> None:
        """Initialize prompt classifier and projection after encoder is loaded."""
        encoder_hidden = self._get_encoder_hidden_dim()
        num_classes = self._registry.total_tokens

        self._prompt_classifier = nn.Linear(encoder_hidden, num_classes)
        logger.info(
            f"Prompt classifier initialized: Linear({encoder_hidden}, {num_classes})"
        )

        # Add projection if prompt embed dim != encoder input dim
        embed_dim = self._multitask_cfg.prompt_embed_dim
        encoder_input_dim = self._get_encoder_input_dim()
        if embed_dim != encoder_input_dim:
            self._prompt_projection = nn.Linear(embed_dim, encoder_input_dim)
            logger.info(
                f"Prompt projection added: Linear({embed_dim}, {encoder_input_dim})"
            )

    def _get_encoder_hidden_dim(self) -> int:
        """Get encoder output dimension from loaded NeMo model."""
        asr = self._base_model.asr_model
        if hasattr(asr, "encoder") and hasattr(asr.encoder, "_feat_out"):
            return asr.encoder._feat_out
        if hasattr(asr, "cfg") and "encoder" in asr.cfg:
            enc_cfg = asr.cfg.encoder
            if hasattr(enc_cfg, "d_model"):
                return enc_cfg.d_model
        # Fallback: try model_defaults
        if hasattr(asr, "cfg") and "model_defaults" in asr.cfg:
            if hasattr(asr.cfg.model_defaults, "enc_hidden"):
                return asr.cfg.model_defaults.enc_hidden
        raise RuntimeError("Cannot determine encoder hidden dim from loaded model")

    def _get_encoder_input_dim(self) -> int:
        """Get encoder input dimension (typically fbank dim or after pre-encoder)."""
        asr = self._base_model.asr_model
        if hasattr(asr, "cfg") and "preprocessor" in asr.cfg:
            if hasattr(asr.cfg.preprocessor, "features"):
                return asr.cfg.preprocessor.features
        return self._multitask_cfg.prompt_embed_dim  # Default: assume match

    def _prepend_prompts(
        self,
        speech: torch.Tensor,
        task_labels: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        """Prepend prompt embeddings to speech features.

        Args:
            speech: [B, T, D] speech features (post-preprocessor)
            task_labels: {task_name: Tensor[B]} class indices

        Returns:
            (augmented_speech [B, T+num_prompts, D], num_prepended)
        """
        prompt_embeds = self._prompt_embed(task_labels)  # [B, 6, embed_dim]

        if self._prompt_projection is not None:
            prompt_embeds = self._prompt_projection(prompt_embeds)

        augmented = torch.cat([prompt_embeds, speech], dim=1)
        return augmented, self._prompt_embed.num_positions

    def _compute_prompt_loss(
        self,
        encoder_out: torch.Tensor,
        task_labels: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute CE loss on prompt positions of encoder output.

        Reference: funasr/models/sense_voice/model.py:793-807

        Args:
            encoder_out: [B, T'+num_prompts, H] full encoder output
            task_labels: {task_name: Tensor[B]} ground truth class indices

        Returns:
            (loss, accuracy_dict) where accuracy_dict has per-task accuracy
        """
        num_prompts = self._prompt_embed.num_positions
        prompt_out = encoder_out[:, :num_prompts, :]  # [B, 6, H]
        logits = self._prompt_classifier(prompt_out)  # [B, 6, total_tokens]

        total_loss = torch.tensor(0.0, device=encoder_out.device)
        accuracy_dict = {}

        for i, task in enumerate(self._multitask_cfg.task_order):
            task_logits = logits[:, i, :]  # [B, total_tokens]

            if task in task_labels:
                targets = task_labels[task]  # [B]
            else:
                targets = torch.zeros(
                    encoder_out.shape[0], dtype=torch.long, device=encoder_out.device
                )

            # Shift targets to match embedding ID space
            start, _ = self._registry.get_task_range(task)
            targets_shifted = targets + start

            loss_i = self._prompt_loss_fn(task_logits, targets_shifted)
            total_loss = total_loss + loss_i

            # Compute accuracy
            with torch.no_grad():
                preds = task_logits.argmax(dim=-1)
                correct = (preds == targets_shifted).float().mean().item()
                accuracy_dict[f"acc_{task}"] = correct

        total_loss = total_loss / num_prompts  # Average across positions
        return total_loss, accuracy_dict

    def predict_prompt_labels(
        self, encoder_out: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Decode prompt positions to paralinguistic labels (inference).

        Args:
            encoder_out: [B, T'+num_prompts, H]

        Returns:
            {task_name: Tensor[B]} with predicted class indices
        """
        num_prompts = self._prompt_embed.num_positions
        prompt_out = encoder_out[:, :num_prompts, :]
        logits = self._prompt_classifier(prompt_out)  # [B, 6, total_tokens]

        predictions = {}
        for i, task in enumerate(self._multitask_cfg.task_order):
            task_logits = logits[:, i, :]  # [B, total_tokens]
            start, end = self._registry.get_task_range(task)
            # Restrict prediction to valid range for this task
            task_specific = task_logits[:, start:end]  # [B, num_classes_for_task]
            predictions[task] = task_specific.argmax(dim=-1)  # [B]

        return predictions

    def forward_multitask(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        task_labels: Optional[Dict[str, torch.Tensor]] = None,
        text: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Full multi-task forward pass.

        Args:
            audio: [B, T] raw audio samples
            audio_lengths: [B]
            task_labels: {task_name: Tensor[B]} for prompt CE loss (None = inference)
            text: [B, L] token IDs for RNNT loss (None = inference)
            text_lengths: [B]

        Returns:
            Dict with: loss_total, loss_rnnt, loss_prompt, prompt_preds, accuracy
        """
        asr = self._base_model.asr_model

        # 1. Preprocessor: audio → features
        processed, processed_lens = asr.preprocessor(
            input_signal=audio, length=audio_lengths
        )

        # 2. Spec augmentation (training only)
        if asr.training and hasattr(asr, "spec_augmentation") and asr.spec_augmentation is not None:
            processed = asr.spec_augmentation(input_spec=processed, length=processed_lens)

        # 3. Prepend prompt embeddings
        if task_labels is not None:
            augmented, num_prepended = self._prepend_prompts(processed, task_labels)
            augmented_lens = processed_lens + num_prepended
        else:
            augmented = processed
            augmented_lens = processed_lens
            num_prepended = 0

        # 4. Encode
        encoded, encoded_lens = asr.encoder(
            audio_signal=augmented, length=augmented_lens
        )

        result = {}

        # 5. Prompt CE loss (training with task labels)
        if task_labels is not None and num_prepended > 0:
            loss_prompt, accuracy = self._compute_prompt_loss(encoded, task_labels)
            result["loss_prompt"] = loss_prompt
            result["accuracy"] = accuracy
        else:
            loss_prompt = torch.tensor(0.0, device=audio.device)
            result["loss_prompt"] = loss_prompt
            result["accuracy"] = {}

        # 6. Predict prompt labels (always available after encoding)
        if num_prepended > 0:
            result["prompt_preds"] = self.predict_prompt_labels(encoded)

        # 7. RNNT loss (training with text)
        if text is not None and text_lengths is not None:
            # Strip prompt positions for RNNT decoder
            if num_prepended > 0:
                speech_encoded = encoded[:, num_prepended:, :]
                speech_encoded_lens = encoded_lens - num_prepended
            else:
                speech_encoded = encoded
                speech_encoded_lens = encoded_lens

            decoder_out, target_len, states = asr.decoder(
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
            loss_rnnt = torch.tensor(0.0, device=audio.device)
            result["loss_rnnt"] = loss_rnnt

        # 8. Combine losses
        total_loss, loss_stats = self._multi_task_loss(loss_rnnt, loss_prompt)
        result["loss_total"] = total_loss
        result["loss_stats"] = loss_stats

        return result

    def freeze_encoder_layers(self, num_layers: int) -> None:
        """Freeze the first N encoder layers."""
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
        """Unfreeze all parameters."""
        for param in self._base_model.asr_model.parameters():
            param.requires_grad = True
        for param in self._prompt_embed.parameters():
            param.requires_grad = True
        if self._prompt_classifier is not None:
            for param in self._prompt_classifier.parameters():
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
        """Load encoder weights from a FunASR checkpoint into the NeMo encoder.

        Maps FunASR 'audio_encoder.*' keys to NeMo 'encoder.*' keys.
        """
        state_dict = torch.load(funasr_checkpoint_path, map_location="cpu")

        # Extract encoder keys
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
