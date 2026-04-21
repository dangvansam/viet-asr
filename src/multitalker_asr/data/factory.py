from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from ..models.prompt_embedding import TaskTokenRegistry
from .datasets.streaming import StreamingMultitalkerDataset
from .datasets.multitask import MultitaskStreamingDataset
from .collators.multitalker import MultitalkerCollator
from .collators.multitask import MultitaskCollator


class DataLoaderFactory:
    @staticmethod
    def create_streaming_dataloader(
        manifest_paths: List[str],
        tokenizer=None,
        batch_size: int = 16,
        max_speakers: int = 2,
        num_workers: int = 4,
        seed: int = 42,
        max_samples: Optional[int] = None,
        pin_memory: bool = True,
    ) -> DataLoader:
        dataset = StreamingMultitalkerDataset(
            manifest_paths=manifest_paths,
            tokenizer=tokenizer,
            max_speakers=max_speakers,
            seed=seed,
            max_samples=max_samples,
        )

        collator = MultitalkerCollator(tokenizer=tokenizer)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
        )

    @staticmethod
    def create_train_dataloader(
        manifest_paths: List[str],
        tokenizer=None,
        batch_size: int = 16,
        max_speakers: int = 2,
        num_workers: int = 4,
    ) -> DataLoader:
        return DataLoaderFactory.create_streaming_dataloader(
            manifest_paths=manifest_paths,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_speakers=max_speakers,
            num_workers=num_workers,
            max_samples=None,
        )

    @staticmethod
    def create_val_dataloader(
        manifest_paths: List[str],
        tokenizer=None,
        batch_size: int = 16,
        max_speakers: int = 2,
        num_workers: int = 4,
        max_samples: int = 320,
    ) -> DataLoader:
        return DataLoaderFactory.create_streaming_dataloader(
            manifest_paths=manifest_paths,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_speakers=max_speakers,
            num_workers=num_workers,
            max_samples=max_samples,
        )

    @staticmethod
    def create_multitask_dataloader(
        manifest_paths: List[str],
        tokenizer=None,
        task_registry: Optional[TaskTokenRegistry] = None,
        batch_size: int = 16,
        max_speakers: int = 2,
        num_workers: int = 4,
        seed: int = 42,
        max_samples: Optional[int] = None,
        default_labels: Optional[Dict[str, str]] = None,
        pin_memory: bool = True,
    ) -> DataLoader:
        dataset = MultitaskStreamingDataset(
            manifest_paths=manifest_paths,
            tokenizer=tokenizer,
            task_registry=task_registry,
            max_speakers=max_speakers,
            seed=seed,
            max_samples=max_samples,
            default_labels=default_labels,
        )

        collator = MultitaskCollator(tokenizer=tokenizer)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
        )
