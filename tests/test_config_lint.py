"""Regression guard: pipeline configs must run ASR via local OpenAI services only.

Every `multi_transcribe` / `transcript_refine` backend must be the unified
`openai_transcription` client (or `google_speech`, the one allowed cloud backend) —
no in-venv ASR. Every audio `transcribe` stage must be in service mode (`base_url`).
"""

import glob
import os

from omegaconf import OmegaConf

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
ALLOWED_ASR = {"openai_transcription", "openai", "google_speech"}
IN_VENV_ASR = {"funasr", "nemotron", "vietasr", "qwen3", "funasr_mlt", "qwen3_vllm", "service"}


def _configs():
    return sorted(glob.glob(os.path.join(CONFIG_DIR, "pipeline_*.yaml")))


def _load(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)


def _asr_backend_lists(cfg):
    sc = cfg.get("stage_configs") or {}
    for key in ("multi_transcribe", "transcript_refine"):
        stage = sc.get(key)
        if isinstance(stage, dict) and isinstance(stage.get("backends"), list):
            yield key, stage["backends"]


def test_configs_found():
    assert _configs(), "no pipeline_*.yaml configs found"


def test_no_in_venv_asr_backends():
    offenders = []
    for path in _configs():
        cfg = _load(path)
        for stage, backends in _asr_backend_lists(cfg):
            for b in backends:
                name = b.get("name") if isinstance(b, dict) else None
                if name in IN_VENV_ASR:
                    offenders.append(f"{os.path.basename(path)}:{stage} -> {name}")
                elif name is not None and name not in ALLOWED_ASR:
                    offenders.append(f"{os.path.basename(path)}:{stage} -> {name}")
    assert not offenders, "non-service ASR backends in configs: " + ", ".join(offenders)


def test_transcribe_stage_uses_service():
    offenders = []
    for path in _configs():
        cfg = _load(path)
        for block in (cfg.get("transcribe"), (cfg.get("stage_configs") or {}).get("transcribe")):
            if not isinstance(block, dict):
                continue
            if block.get("itn_only"):  # text-only normalization, not audio ASR
                continue
            if not block.get("base_url"):
                offenders.append(os.path.basename(path))
    assert not offenders, "transcribe stage not in service mode (missing base_url): " + ", ".join(offenders)
