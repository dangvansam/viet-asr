import random
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from lhotse import SupervisionSegment
from loguru import logger

from .base import BaseMixer


class MultiTalkerMixer(BaseMixer):
    def __init__(self, sr_target: int = 16000):
        self._sr_target = sr_target

    @property
    def sample_rate(self) -> int:
        return self._sr_target

    def mix(
        self,
        utterances: List[Dict[str, Any]],
        sample_id: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[List[SupervisionSegment]], float]:
        timeline = []
        current_time = 0.0

        for ut in utterances:
            audio_data = self._load_audio(ut)
            if audio_data is None:
                continue

            start_t = random.uniform(0.0, max(0.0, current_time * 0.7))
            timeline.append((start_t, audio_data, ut))
            current_time = max(current_time, start_t + (len(audio_data) / self._sr_target))

        if not timeline:
            return None, None, 0.0

        total_duration = max(
            start_t + (len(y) / self._sr_target) for start_t, y, _ in timeline
        )
        mixed_audio = np.zeros(int(total_duration * self._sr_target) + 1)
        supervisions = []
        mix_id = sample_id or f"mixed_{random.randint(0, 100000):05d}"

        for idx, (start_t, audio_data, ut) in enumerate(timeline):
            start_sample = int(start_t * self._sr_target)
            end_sample = start_sample + len(audio_data)
            mixed_audio[start_sample:end_sample] += audio_data

            supervisions.append(
                SupervisionSegment(
                    id=f"{mix_id}-sup{idx}",
                    recording_id=mix_id,
                    start=round(start_t, 4),
                    duration=round(len(audio_data) / self._sr_target, 4),
                    channel=0,
                    speaker=str(ut.get("label", ut.get("speaker", f"speaker_{idx}"))),
                    text=str(ut["text"]),
                )
            )

        max_amp = np.max(np.abs(mixed_audio))
        if max_amp > 1.0:
            mixed_audio /= max_amp

        return mixed_audio, supervisions, total_duration

    def mix_utterances(
        self,
        utterances: List[Dict[str, Any]],
        sample_id: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[List[SupervisionSegment]], float]:
        return self.mix(utterances, sample_id)

    def _load_audio(self, utterance: Dict[str, Any]) -> Optional[np.ndarray]:
        try:
            audio, sr = sf.read(utterance["audio_filepath"], dtype="float32")

            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            offset = utterance.get("offset", 0.0)
            duration = utterance.get("duration", None)

            if duration:
                start_sample = int(offset * sr)
                end_sample = start_sample + int(duration * sr)
                audio = audio[start_sample:end_sample]

            if sr != self._sr_target:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self._sr_target)

            return audio
        except Exception as e:
            logger.error(f"Error loading {utterance.get('audio_filepath')}: {e}")
            return None
