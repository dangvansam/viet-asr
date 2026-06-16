import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import BaseConfig


SUPPORTED_MODELS = ("chirp_3", "chirp_2")

MODEL_LOCATIONS = {
    "chirp_3": "us",
    "chirp_2": "us-central1",
}


@dataclass
class GoogleSpeechConfig(BaseConfig):
    project_id: Optional[str] = None
    location: Optional[str] = None
    recognizer: str = "_"
    credentials_path: Optional[str] = None
    project_id_env: str = "GOOGLE_CLOUD_PROJECT"

    model: str = "chirp_3"
    language_codes: List[str] = field(default_factory=lambda: ["vi-VN"])

    enable_automatic_punctuation: bool = True
    enable_word_time_offsets: bool = True
    enable_word_confidence: bool = True
    enable_spoken_punctuation: bool = False
    profanity_filter: bool = False
    max_alternatives: int = 1

    phrase_sets: List[Dict[str, Any]] = field(default_factory=list)

    batch_timeout: float = 600.0

    def __post_init__(self):
        if not self.project_id and self.project_id_env:
            self.project_id = os.environ.get(self.project_id_env)
        if not self.project_id:
            raise ValueError(
                "GoogleSpeechConfig.project_id is empty and could not be resolved "
                f"from env '{self.project_id_env}'"
            )
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(
                f"GoogleSpeechConfig.model must be one of {SUPPORTED_MODELS}, "
                f"got '{self.model}'"
            )
        if not self.location:
            self.location = MODEL_LOCATIONS[self.model]
        if not self.recognizer:
            raise ValueError("GoogleSpeechConfig.recognizer must be non-empty")
        if not self.language_codes:
            raise ValueError("GoogleSpeechConfig.language_codes must be non-empty")
        if self.max_alternatives < 1:
            raise ValueError(
                f"max_alternatives must be >= 1, got {self.max_alternatives}"
            )
        if self.batch_timeout <= 0.0:
            raise ValueError(f"batch_timeout must be > 0, got {self.batch_timeout}")

    @property
    def recognizer_path(self) -> str:
        return (
            f"projects/{self.project_id}/locations/{self.location}"
            f"/recognizers/{self.recognizer}"
        )
