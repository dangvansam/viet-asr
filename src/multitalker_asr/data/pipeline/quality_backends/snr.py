import numpy as np

from .base import BaseQualityBackend, QualityResult

_EPS = 1e-10


class SNREstimatorBackend(BaseQualityBackend):
    """Lightweight energy-based SNR estimate (no model).

    Frames the signal, takes a low percentile of frame power as the noise floor
    and a high percentile as the speech level; SNR(dB) = 10*log10(speech/noise).
    A normalized score in [0, 1] maps 0..`score_ceiling_db` dB linearly.
    """

    name = "snr"

    def __init__(
        self,
        frame_ms: float = 25.0,
        hop_ms: float = 10.0,
        noise_percentile: float = 10.0,
        speech_percentile: float = 90.0,
        score_ceiling_db: float = 40.0,
    ):
        self._frame_ms = frame_ms
        self._hop_ms = hop_ms
        self._noise_pct = noise_percentile
        self._speech_pct = speech_percentile
        self._ceiling = score_ceiling_db
        self._loaded = True

    def load(self, device: str = "cpu") -> None:
        self._loaded = True

    def score(self, audio: np.ndarray, sample_rate: int) -> QualityResult:
        x = np.asarray(audio, dtype=np.float64).reshape(-1)
        powers = self._frame_powers(x, sample_rate)
        if powers.size == 0:
            return QualityResult(snr_db=0.0, score=0.0, backend=self.name)

        noise = float(np.percentile(powers, self._noise_pct))
        speech = float(np.percentile(powers, self._speech_pct))
        snr_db = 10.0 * np.log10((speech + _EPS) / (noise + _EPS))
        snr_db = float(max(0.0, snr_db))
        score = float(np.clip(snr_db / self._ceiling, 0.0, 1.0))
        return QualityResult(
            snr_db=round(snr_db, 2),
            score=round(score, 4),
            backend=self.name,
            raw={"noise_pow": noise, "speech_pow": speech, "n_frames": int(powers.size)},
        )

    def _frame_powers(self, x: np.ndarray, sr: int) -> np.ndarray:
        frame = max(1, int(self._frame_ms * sr / 1000.0))
        hop = max(1, int(self._hop_ms * sr / 1000.0))
        if x.size < frame:
            return np.array([float(np.mean(x ** 2))]) if x.size else np.array([])
        n = 1 + (x.size - frame) // hop
        idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
        frames = x[idx]
        return np.mean(frames ** 2, axis=1)

    def unload(self) -> None:
        return None
