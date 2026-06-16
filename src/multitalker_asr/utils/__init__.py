"""Utils public API.

Lazy (PEP 562) so the torch-free helpers (AudioLoader, TextExtractor, format
converters) used by the lightweight pipeline image are importable without pulling
in torch via device.py / checkpoint.py.
"""

import importlib

_LAZY = {
    "AudioLoader": "audio",
    "CheckpointManager": "checkpoint",
    "DeviceManager": "device",
    "RTTMConverter": "formats",
    "AudacityConverter": "formats",
    "PyannoteConverter": "formats",
    "LLMClient": "llm_client",
    "LLMClientError": "llm_client",
    "LLMClientFactory": "llm_client",
    "LLMResponse": "llm_client",
    "TextExtractor": "text",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module 'multitalker_asr.utils' has no attribute '{name}'")
    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)


def __dir__():
    return sorted(__all__)
