from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

import torch
from loguru import logger

from ..configs import InferenceConfig
from ..configs.multitask import MultiTaskConfig
from ..models.multitask_model import MultitalkerMultiTaskModel
from ..models.prompt_embedding import TaskTokenRegistry
from .base import BaseInferenceEngine
from .post_processor import FunASRPostProcessor


@dataclass
class SpeakerResult:
    """Result for a single speaker from multi-task inference."""

    speaker_id: str = ""
    text: str = ""
    text_refined: Optional[str] = None
    emotion: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[str] = None
    voice_state: Optional[str] = None
    language: Optional[str] = None
    eou_detected: bool = False
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        parts = [f"[{self.speaker_id}]"]
        if self.text_refined:
            parts.append(self.text_refined)
        else:
            parts.append(self.text)
        labels = []
        if self.emotion:
            labels.append(f"emotion={self.emotion}")
        if self.gender:
            labels.append(f"gender={self.gender}")
        if self.age:
            labels.append(f"age={self.age}")
        if self.voice_state:
            labels.append(f"voice_state={self.voice_state}")
        if self.language:
            labels.append(f"lang={self.language}")
        if labels:
            parts.append(f"({', '.join(labels)})")
        if self.eou_detected:
            parts.append("[EOU]")
        return " ".join(parts)


class MultitaskInferenceEngine(BaseInferenceEngine):
    """End-to-end multi-task inference with optional FunASR post-processing."""

    def __init__(
        self,
        model: MultitalkerMultiTaskModel,
        inference_cfg: Optional[InferenceConfig] = None,
        multitask_cfg: Optional[MultiTaskConfig] = None,
        post_processor: Optional[FunASRPostProcessor] = None,
    ):
        self._model = model
        self._inference_cfg = inference_cfg or InferenceConfig()
        self._multitask_cfg = multitask_cfg or MultiTaskConfig()
        self._post_processor = post_processor
        self._registry = model.registry if model else TaskTokenRegistry(self._multitask_cfg)

    def setup(self) -> None:
        """No-op setup — model is passed pre-loaded."""
        pass

    def infer(self, audio_path: str) -> List[SpeakerResult]:
        """Full inference pipeline: transcribe + classify + optionally refine.

        Args:
            audio_path: Path to audio file

        Returns:
            List of SpeakerResult, one per detected speaker
        """
        import soundfile as sf

        audio_data, sr = sf.read(audio_path, dtype="float32")
        if sr != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
        audio_lengths = torch.tensor([audio_tensor.shape[1]])

        device = self._model.device
        audio_tensor = audio_tensor.to(device)
        audio_lengths = audio_lengths.to(device)

        # Forward pass (inference mode — no task labels, no text)
        with torch.no_grad():
            result = self._model.forward_multitask(
                audio=audio_tensor,
                audio_lengths=audio_lengths,
            )

        # Build speaker result
        speaker_result = SpeakerResult(speaker_id="speaker_0")

        # Decode prompt predictions
        if "prompt_preds" in result:
            preds = result["prompt_preds"]
            speaker_result = self._decode_labels(speaker_result, preds)

        # Optionally refine with FunASR
        if self._post_processor is not None:
            refined = self._post_processor.refine_text(audio_path=audio_path)
            speaker_result.text_refined = refined
            if not speaker_result.text:
                speaker_result.text = refined

        return [speaker_result]

    def _decode_labels(
        self, result: SpeakerResult, preds: Dict[str, torch.Tensor]
    ) -> SpeakerResult:
        """Map predicted class indices to human-readable labels."""
        for task, pred_tensor in preds.items():
            idx = pred_tensor[0].item()
            label = self._registry.get_label_name(task, idx)
            if task == "emotion":
                result.emotion = label
            elif task == "gender":
                result.gender = label
            elif task == "age":
                result.age = label
            elif task == "voice_state":
                result.voice_state = label
            elif task == "language":
                result.language = label
        return result

    @staticmethod
    def detect_eou(token_ids: List[int], eou_token_id: int) -> bool:
        """Check if <EOU> token appears in decoded token sequence."""
        return eou_token_id in token_ids
