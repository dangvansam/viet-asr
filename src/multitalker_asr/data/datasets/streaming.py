import json
import random
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
from loguru import logger

from ..base import BaseDataset
from ..mixers.multitalker import MultiTalkerMixer


class StreamingMultitalkerDataset(BaseDataset, torch.utils.data.IterableDataset):
    def __init__(
        self,
        manifest_paths: List[str],
        tokenizer=None,
        mixer: Optional[MultiTalkerMixer] = None,
        max_speakers: int = 2,
        seed: int = 42,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self._manifest_paths = manifest_paths
        self._tokenizer = tokenizer
        self._mixer = mixer or MultiTalkerMixer()
        self._max_speakers = max_speakers
        self._seed = seed
        self._max_samples = max_samples
        self._utterances = self._load_utterances()

        logger.info(f"Loaded {len(self._utterances)} source utterances for streaming.")

    def _load_utterances(self) -> List[Dict[str, Any]]:
        utterances = []
        for path in self._manifest_paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    utterances.append(json.loads(line))
        return utterances

    def __iter__(self) -> Iterator[Dict[str, Any]]:
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

            yield {
                "audio": audio,
                "audio_len": len(audio),
                "text": combined_text,
                "text_ids": text_ids,
                "duration": duration,
                "num_speakers": num_spk,
                "spk_mask": spk_mask[: len(audio)],
                "bg_mask": bg_mask[: len(audio)],
            }
