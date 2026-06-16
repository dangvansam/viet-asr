"""Public API.

Exposed names are loaded lazily (PEP 562) so importing a submodule — e.g.
`multitalker_asr.data.pipeline.vad_backends` in the lightweight, URL-only pipeline
image — does NOT pull in torch/NeMo via the model classes. Heavy modules are only
imported when their symbol is actually accessed.
"""

import importlib

_LAZY = {
    "ModelConfig": "configs",
    "TrainingConfig": "configs",
    "InferenceConfig": "configs",
    "DataConfig": "configs",
    "EvalConfig": "configs",
    "MultiTaskConfig": "configs",
    "BaseASRModel": "models",
    "MultitalkerASRModel": "models",
    "MultitalkerMultiTaskModel": "models",
    "TokenizerExtender": "models",
    "TaskTokenRegistry": "models",
    "PromptEmbedding": "models",
    "MultiTaskLoss": "training.losses",
    "BaseHead": "models.heads",
    "SpeakerHead": "models.heads",
    "GenderHead": "models.heads",
    "EmotionHead": "models.heads",
    "AgeHead": "models.heads",
    "DataLoaderFactory": "data",
    "MultitalkerCollator": "data",
    "MultiTalkerMixer": "data",
    "MultitalkerSynthesizer": "data",
    "StreamingMultitalkerDataset": "data",
    "ManifestReader": "data",
    "ManifestWriter": "data",
    "BaseTrainer": "training",
    "MultitalkerTrainer": "training",
    "BaseInferenceEngine": "inference",
    "StreamingInferenceEngine": "inference",
    "OfflineInferenceEngine": "inference",
    "Transcriber": "inference",
    "DiarizationEvaluator": "eval",
    "EvaluationPipeline": "eval",
    "EvalDataSynthesizer": "eval",
    "DERMetric": "eval",
    "LatencyMetric": "eval",
    "TextReporter": "eval",
    "ChartReporter": "eval",
    "DeviceManager": "utils",
    "CheckpointManager": "utils",
    "TextExtractor": "utils",
    "AudioLoader": "utils",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module 'multitalker_asr' has no attribute '{name}'")
    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)


def __dir__():
    return sorted(__all__)
