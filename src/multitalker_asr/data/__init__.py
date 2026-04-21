from .base import BaseDataset, BaseCollator, BaseMixer
from .datasets import StreamingMultitalkerDataset
from .datasets.multitask import MultitaskStreamingDataset
from .collators import MultitalkerCollator
from .collators.multitask import MultitaskCollator
from .mixers import MultiTalkerMixer
from .synthesizers import MultitalkerSynthesizer
from .manifest import ManifestReader, ManifestWriter
from .factory import DataLoaderFactory

__all__ = [
    "BaseDataset",
    "BaseCollator",
    "BaseMixer",
    "StreamingMultitalkerDataset",
    "MultitaskStreamingDataset",
    "MultitalkerCollator",
    "MultitaskCollator",
    "MultiTalkerMixer",
    "MultitalkerSynthesizer",
    "ManifestReader",
    "ManifestWriter",
    "DataLoaderFactory",
]


def get_multitalker_dataloader(
    manifest_paths,
    tokenizer=None,
    batch_size=16,
    max_speakers=2,
    num_workers=4,
    seed=42,
    max_samples=None,
):
    return DataLoaderFactory.create_streaming_dataloader(
        manifest_paths=manifest_paths,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_speakers=max_speakers,
        num_workers=num_workers,
        seed=seed,
        max_samples=max_samples,
    )
