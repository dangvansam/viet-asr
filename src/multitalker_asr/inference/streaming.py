from typing import Any, Dict, List, Optional

import torch
from loguru import logger
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from omegaconf import OmegaConf

from ..configs import InferenceConfig
from .base import BaseInferenceEngine


class StreamingInferenceEngine(BaseInferenceEngine):
    def __init__(
        self,
        asr_model,
        diar_model,
        inference_cfg: Optional[InferenceConfig] = None,
    ):
        self._asr_model = asr_model
        self._diar_model = diar_model
        self._cfg = inference_cfg or InferenceConfig()
        self._streamer = None

    def setup(self) -> None:
        self._setup_diarization_streaming()
        self._setup_asr_streaming()

    def infer(self, audio_path: str) -> List[Dict[str, Any]]:
        if self._streamer is None:
            self.setup()
            self._create_streamer()

        samples = [{"audio_filepath": audio_path}]

        streaming_buffer = CacheAwareStreamingAudioBuffer(
            model=self._asr_model,
            online_normalization=False,
            pad_and_drop_preencoded=False,
        )
        streaming_buffer.append_audio_file(audio_filepath=audio_path, stream_id=-1)

        autocast = torch.amp.autocast(self._asr_model.device.type, enabled=True)

        for step_num, (chunk_audio, chunk_lengths) in enumerate(iter(streaming_buffer)):
            drop_extra = (
                0
                if step_num == 0
                else self._asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
            )
            with torch.inference_mode():
                with autocast:
                    self._streamer.perform_parallel_streaming_stt_spk(
                        step_num=step_num,
                        chunk_audio=chunk_audio,
                        chunk_lengths=chunk_lengths,
                        is_buffer_empty=streaming_buffer.is_buffer_empty(),
                        drop_extra_pre_encoded=drop_extra,
                    )

        return self._streamer.generate_seglst_dicts_from_parallel_streaming(
            samples=samples
        )

    def _setup_diarization_streaming(self) -> None:
        OmegaConf.set_struct(self._diar_model.cfg, False)
        self._diar_model.cfg.stream_params = OmegaConf.create(
            {
                "window_length_s": 0.5,
                "shift_length_s": 0.05,
                "margin_frames": 10,
                "latency_s": 0.5,
            }
        )

        if hasattr(self._diar_model, "sortformer_modules"):
            self._diar_model.sortformer_modules.chunk_len = 0
            self._diar_model.sortformer_modules.spkcache_len = 188
            self._diar_model.sortformer_modules.fifo_len = 188

    def _setup_asr_streaming(self) -> None:
        if self._cfg.att_context_size and hasattr(
            self._asr_model.encoder, "set_default_att_context_size"
        ):
            self._asr_model.encoder.set_default_att_context_size(
                att_context_size=self._cfg.att_context_size
            )

    def _create_streamer(self) -> None:
        from nemo.collections.asr.parts.utils.multispk_transcribe_utils import (
            SpeakerTaggedASR,
        )

        nemo_cfg = OmegaConf.create(
            OmegaConf.to_container(OmegaConf.structured(self._cfg), resolve=True)
        )
        self._streamer = SpeakerTaggedASR(nemo_cfg, self._asr_model, self._diar_model)
