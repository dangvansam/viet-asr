from typing import Any, Dict, List, Optional

from loguru import logger
from nemo.collections.asr.parts.utils.multispk_transcribe_utils import write_seglst_file

from ..configs import InferenceConfig
from ..models import MultitalkerASRModel
from .base import BaseInferenceEngine
from .streaming import StreamingInferenceEngine


class Transcriber:
    def __init__(
        self,
        model: MultitalkerASRModel,
        inference_cfg: Optional[InferenceConfig] = None,
        engine: Optional[BaseInferenceEngine] = None,
    ):
        self._model = model
        self._cfg = inference_cfg or InferenceConfig()
        self._engine = engine

    @property
    def engine(self) -> BaseInferenceEngine:
        if self._engine is None:
            self._engine = StreamingInferenceEngine(
                asr_model=self._model.asr_model,
                diar_model=self._model.diar_model,
                inference_cfg=self._cfg,
            )
        return self._engine

    def transcribe(
        self,
        audio_path: str,
        output_path: Optional[str] = "output.json",
    ) -> List[Dict[str, Any]]:
        results = self.engine.infer(audio_path)

        if output_path and results:
            self._save_results(results, output_path)

        return results

    def transcribe_batch(
        self,
        audio_paths: List[str],
        output_dir: Optional[str] = None,
    ) -> List[List[Dict[str, Any]]]:
        all_results = []
        for audio_path in audio_paths:
            output_path = None
            if output_dir:
                import os
                stem = os.path.splitext(os.path.basename(audio_path))[0]
                output_path = os.path.join(output_dir, f"{stem}.json")

            results = self.transcribe(audio_path, output_path=output_path)
            all_results.append(results)

        return all_results

    def _save_results(
        self,
        results: List[Dict[str, Any]],
        output_path: str,
    ) -> None:
        try:
            write_seglst_file(seglst_dict_list=results, output_path=output_path)
            logger.success(f"Transcription saved to {output_path}")
        except ValueError as e:
            logger.warning(f"Failed to write transcript: {e}")
