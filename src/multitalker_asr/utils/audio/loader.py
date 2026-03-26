from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


class AudioLoader:
    def __init__(self, target_sample_rate: int = 16000):
        self._target_sr = target_sample_rate

    @property
    def sample_rate(self) -> int:
        return self._target_sr

    def load(
        self,
        audio_path: Union[str, Path],
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        import soundfile as sf

        audio_path = str(audio_path)

        if offset > 0 or duration is not None:
            info = sf.info(audio_path)
            start_frame = int(offset * info.samplerate)
            frames = int(duration * info.samplerate) if duration else -1
            audio, sr = sf.read(
                audio_path,
                start=start_frame,
                frames=frames,
                dtype="float32",
            )
        else:
            audio, sr = sf.read(audio_path, dtype="float32")

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        if sr != self._target_sr:
            audio = self._resample(audio, sr, self._target_sr)

        return audio, self._target_sr

    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        import librosa

        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    def load_segment(
        self,
        audio_path: Union[str, Path],
        start: float,
        end: float,
    ) -> Tuple[np.ndarray, int]:
        duration = end - start
        return self.load(audio_path, offset=start, duration=duration)

    def get_duration(self, audio_path: Union[str, Path]) -> float:
        import soundfile as sf

        info = sf.info(str(audio_path))
        return info.duration

    def save(
        self,
        audio: np.ndarray,
        output_path: Union[str, Path],
        sample_rate: Optional[int] = None,
    ) -> None:
        import soundfile as sf

        sr = sample_rate or self._target_sr
        sf.write(str(output_path), audio, sr)
