from .qwen3_service import Qwen3ServiceAlignBackend


class ServiceAlignBackend(Qwen3ServiceAlignBackend):
    """URL-only forced-alignment client for any in-repo align service
    (scripts/serve_align.py: mms_fa / nemo_nfa). Shares the `POST /align`
    contract with Qwen3ServiceAlignBackend, so one client serves every aligner —
    only `base_url` changes. Keeps the pipeline venv free of torchaudio/NeMo."""

    name = "service"
