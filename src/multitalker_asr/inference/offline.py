from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from ..configs import InferenceConfig
from .base import BaseInferenceEngine


class OfflineInferenceEngine(BaseInferenceEngine):
    def __init__(
        self,
        asr_model,
        diar_model,
        inference_cfg: Optional[InferenceConfig] = None,
    ):
        self._asr_model = asr_model
        self._diar_model = diar_model
        self._cfg = inference_cfg or InferenceConfig()

    def setup(self) -> None:
        self._asr_model.eval()
        self._diar_model.eval()

    def infer(self, audio_path: str) -> List[Dict[str, Any]]:
        self.setup()

        with torch.inference_mode():
            autocast = torch.amp.autocast(self._asr_model.device.type, enabled=True)
            with autocast:
                hypotheses = self._asr_model.transcribe(
                    [audio_path],
                    batch_size=1,
                )

        results = []
        if hypotheses:
            for hyp in hypotheses:
                if hasattr(hyp, "text"):
                    results.append({"text": hyp.text, "audio_filepath": audio_path})
                else:
                    results.append({"text": str(hyp), "audio_filepath": audio_path})

        return results
