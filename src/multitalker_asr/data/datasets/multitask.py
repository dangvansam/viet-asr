import random
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
from loguru import logger

from ...models.prompt_embedding import TaskTokenRegistry
from .streaming import StreamingMultitalkerDataset


# Default labels when manifest entries lack task fields
DEFAULT_TASK_LABELS = {
    "emotion": "neutral",
    "gender": "male",
    "age": "young",
    "voice_state": "sober",
    "language": "vi",
    "textnorm": "with_itn",
}


class MultitaskStreamingDataset(StreamingMultitalkerDataset):
    """Extends StreamingMultitalkerDataset with paralinguistic label loading.

    Manifest JSONL format (extended):
        {"audio_filepath": "x.wav", "text": "xin chao", "duration": 3.2,
         "emotion": "happy", "gender": "male", "age": "young",
         "voice_state": "sober", "language": "vi"}

    Missing task fields fall back to default_labels.
    """

    def __init__(
        self,
        manifest_paths: List[str],
        tokenizer=None,
        mixer=None,
        task_registry: Optional[TaskTokenRegistry] = None,
        max_speakers: int = 2,
        default_labels: Optional[Dict[str, str]] = None,
        seed: int = 42,
        max_samples: Optional[int] = None,
    ):
        super().__init__(
            manifest_paths=manifest_paths,
            tokenizer=tokenizer,
            mixer=mixer,
            max_speakers=max_speakers,
            seed=seed,
            max_samples=max_samples,
        )
        self._task_registry = task_registry
        self._default_labels = {**DEFAULT_TASK_LABELS, **(default_labels or {})}

    def _parse_task_labels(self, utterance: dict) -> Dict[str, int]:
        """Extract task labels from a manifest entry, map to int via registry."""
        if self._task_registry is None:
            return {}

        labels = {}
        for task in self._task_registry.task_names():
            label_str = utterance.get(task, self._default_labels.get(task))
            if label_str is None:
                labels[task] = 0
                continue

            try:
                labels[task] = self._task_registry.get_class_idx(task, str(label_str))
            except ValueError:
                logger.warning(
                    f"Unknown label '{label_str}' for task '{task}', using default"
                )
                default_str = self._default_labels.get(task)
                if default_str is not None:
                    try:
                        labels[task] = self._task_registry.get_class_idx(task, default_str)
                    except ValueError:
                        labels[task] = 0
                else:
                    labels[task] = 0

        return labels

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Override parent __iter__ to inject task labels from source utterances."""
        import torch

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        rng = random.Random(self._seed + worker_id)
        count = 0

        while self._max_samples is None or count < (self._max_samples // num_workers):
            count += 1
            num_spk = rng.randint(2, self._max_speakers)
            utts = rng.sample(self._utterances, num_spk)

            audio, supervisions, duration = self._mixer.mix(utts)
            if audio is None:
                continue

            combined_text = " ".join([s.text for s in supervisions])

            text_ids = []
            if self._tokenizer is not None:
                text_ids = self._tokenizer.text_to_ids(combined_text)

            total_samples = int(duration * 16000) + 160
            spk_mask = np.zeros(total_samples, dtype=np.float32)
            bg_mask = np.zeros(total_samples, dtype=np.float32)

            for i, s in enumerate(supervisions):
                start_s = int(s.start * 16000)
                end_s = start_s + int(s.duration * 16000)
                if i == 0:
                    spk_mask[start_s:end_s] = 1.0
                else:
                    bg_mask[start_s:end_s] = 1.0

            # Parse task labels from primary speaker (first in arrival order)
            task_labels = self._parse_task_labels(utts[0])

            yield {
                "audio": audio,
                "audio_len": len(audio),
                "text": combined_text,
                "text_ids": text_ids,
                "duration": duration,
                "num_speakers": num_spk,
                "spk_mask": spk_mask[: len(audio)],
                "bg_mask": bg_mask[: len(audio)],
                "task_labels": task_labels,
            }

    def _enrich_with_task_labels(
        self, sample: dict, source_utterances: List[dict]
    ) -> dict:
        """Add task_labels to a sample from the primary speaker's manifest entry."""
        if source_utterances:
            primary = source_utterances[0]
            sample["task_labels"] = self._parse_task_labels(primary)
        else:
            sample["task_labels"] = {
                task: 0 for task in (self._task_registry.task_names() if self._task_registry else [])
            }
        return sample
