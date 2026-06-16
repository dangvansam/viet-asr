"""Data package public API.

Symbols are loaded lazily (PEP 562) so importing the torch-free `data.pipeline`
subpackage (used by the lightweight, URL-only pipeline image) does not pull in the
torch-heavy datasets/collators/factory. Lazy loading also breaks the
factory <-> trainer import cycle that eager top-level imports used to mask.
"""

import importlib

_LAZY = {
    "BaseDataset": "base",
    "BaseCollator": "base",
    "BaseMixer": "base",
    "StreamingMultitalkerDataset": "datasets",
    "MultitaskStreamingDataset": "datasets.multitask",
    "MultitalkerCollator": "collators",
    "MultitaskCollator": "collators.multitask",
    "MultiTalkerMixer": "mixers",
    "MultitalkerSynthesizer": "synthesizers",
    "ManifestReader": "manifest",
    "ManifestWriter": "manifest",
    "SCHEMA_VERSION": "manifest_schema",
    "AttributeConfidence": "manifest_schema",
    "ManifestMigrator": "manifest_schema",
    "ManifestRecord": "manifest_schema",
    "TypedManifestReader": "manifest_schema",
    "TypedManifestWriter": "manifest_schema",
    "DataLoaderFactory": "factory",
}

__all__ = list(_LAZY) + ["get_multitalker_dataloader"]


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module 'multitalker_asr.data' has no attribute '{name}'")
    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)


def __dir__():
    return sorted(__all__)


def get_multitalker_dataloader(
    manifest_paths,
    tokenizer=None,
    batch_size=16,
    max_speakers=2,
    num_workers=4,
    seed=42,
    max_samples=None,
):
    from .factory import DataLoaderFactory

    return DataLoaderFactory.create_streaming_dataloader(
        manifest_paths=manifest_paths,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_speakers=max_speakers,
        num_workers=num_workers,
        seed=seed,
        max_samples=max_samples,
    )
