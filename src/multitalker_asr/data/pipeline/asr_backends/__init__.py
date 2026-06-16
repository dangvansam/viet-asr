from typing import Dict, List, Type

from .base import ASRBackendError, ASRResult, BaseASRBackend, WordTiming
from .ensemble import (
    ASREnsemblerFactory,
    BaseASREnsembler,
    LLMJudgeASREnsembler,
    SingleASREnsembler,
    VoteASREnsembler,
    WordRoverEnsembler,
)
from .funasr import FunASRBackend
from .google_speech import GoogleSpeechBackend
from .nemotron import NemotronStreamingASR
from .openai_transcription import OpenAITranscriptionBackend
from .vietasr import VietASRBackend


# All local ASR runs as a REST service speaking the OpenAI Audio API
# (/v1/audio/transcriptions) — one unified `openai_transcription` client hits them
# all (Qwen3-ASR/Fun-ASR via vLLM, nemotron/vietasr via the serving wrapper). The
# in-venv loader classes (funasr/nemotron/vietasr) run *inside* the service
# containers (scripts/serve_asr.py). google_speech is the only cloud backend.
ASR_REGISTRY: Dict[str, Type[BaseASRBackend]] = {
    "openai_transcription": OpenAITranscriptionBackend,
    "openai": OpenAITranscriptionBackend,
    "google_speech": GoogleSpeechBackend,
    "funasr": FunASRBackend,
    "nemotron": NemotronStreamingASR,
    "vietasr": VietASRBackend,
}


def register_asr_backend(name: str, cls: Type[BaseASRBackend]) -> None:
    ASR_REGISTRY[name] = cls


def build_asr_backend(name: str, **kwargs) -> BaseASRBackend:
    if name not in ASR_REGISTRY:
        raise ValueError(
            f"Unknown ASR backend '{name}'. Registered: {list(ASR_REGISTRY.keys())}"
        )
    return ASR_REGISTRY[name](**kwargs)


def list_asr_backends() -> List[str]:
    return list(ASR_REGISTRY.keys())


__all__ = [
    "ASRBackendError",
    "ASRResult",
    "BaseASRBackend",
    "WordTiming",
    "ASREnsemblerFactory",
    "BaseASREnsembler",
    "SingleASREnsembler",
    "VoteASREnsembler",
    "WordRoverEnsembler",
    "LLMJudgeASREnsembler",
    "FunASRBackend",
    "NemotronStreamingASR",
    "VietASRBackend",
    "GoogleSpeechBackend",
    "OpenAITranscriptionBackend",
    "ASR_REGISTRY",
    "register_asr_backend",
    "build_asr_backend",
    "list_asr_backends",
]
