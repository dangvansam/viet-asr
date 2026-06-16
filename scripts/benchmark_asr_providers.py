import argparse
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import soundfile as sf
from loguru import logger

from multitalker_asr.data.pipeline.asr_backends import build_asr_backend


@dataclass
class ProviderSpec:
    label: str
    backend: str
    kwargs: Dict[str, Any]


@dataclass
class ProviderResult:
    label: str
    backend: str
    ok: bool
    load_seconds: float = 0.0
    transcribe_seconds: float = 0.0
    rtf: float = 0.0
    text: str = ""
    confidence: float = 0.0
    error: str = ""


class ProviderBenchmark:
    def __init__(self, audio_path: str, language: Optional[str] = None):
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        self._audio = audio
        self._sample_rate = sample_rate
        self._duration = len(audio) / sample_rate
        self._language = language
        logger.info(
            f"Loaded {audio_path}: sr={sample_rate} dur={self._duration:.2f}s"
        )

    def run(self, providers: List[ProviderSpec]) -> List[ProviderResult]:
        return [self._run_one(spec) for spec in providers]

    def _run_one(self, spec: ProviderSpec) -> ProviderResult:
        logger.info(f"--- {spec.label} ({spec.backend}) ---")
        try:
            backend = build_asr_backend(spec.backend, **spec.kwargs)
            load_start = time.perf_counter()
            backend.load()
            load_seconds = time.perf_counter() - load_start

            tr_start = time.perf_counter()
            result = backend.transcribe(
                self._audio, self._sample_rate, self._language
            )
            transcribe_seconds = time.perf_counter() - tr_start
            backend.unload()

            rtf = transcribe_seconds / self._duration if self._duration else 0.0
            logger.success(
                f"{spec.label}: {transcribe_seconds:.2f}s rtf={rtf:.3f} "
                f"text='{result.text[:80]}'"
            )
            return ProviderResult(
                label=spec.label,
                backend=spec.backend,
                ok=True,
                load_seconds=load_seconds,
                transcribe_seconds=transcribe_seconds,
                rtf=rtf,
                text=result.text,
                confidence=result.confidence,
            )
        except Exception as exc:
            logger.warning(f"{spec.label} failed: {exc}")
            return ProviderResult(
                label=spec.label, backend=spec.backend, ok=False, error=str(exc)
            )


def print_report(results: List[ProviderResult], duration: float) -> None:
    print("\n" + "=" * 96)
    print(f"ASR PROVIDER BENCHMARK  (audio duration: {duration:.2f}s)")
    print("=" * 96)
    header = f"{'PROVIDER':<22}{'STATUS':<9}{'LOAD(s)':<9}{'INFER(s)':<10}{'RTF':<8}{'CONF':<7}"
    print(header)
    print("-" * 96)
    for r in results:
        status = "OK" if r.ok else "FAIL"
        load = f"{r.load_seconds:.2f}" if r.ok else "-"
        infer = f"{r.transcribe_seconds:.2f}" if r.ok else "-"
        rtf = f"{r.rtf:.3f}" if r.ok else "-"
        conf = f"{r.confidence:.2f}" if r.ok else "-"
        print(f"{r.label:<22}{status:<9}{load:<9}{infer:<10}{rtf:<8}{conf:<7}")
    print("-" * 96)
    print("\nTRANSCRIPTS / ERRORS")
    print("-" * 96)
    for r in results:
        if r.ok:
            print(f"[{r.label}] {r.text!r}")
        else:
            print(f"[{r.label}] ERROR: {r.error}")
    print("=" * 96 + "\n")


def default_providers(language: Optional[str]) -> List[ProviderSpec]:
    lang = language or "vi-VN"
    return [
        ProviderSpec("google-chirp3", "google_speech", {
            "model": "chirp_3", "language_codes": [lang]}),
        ProviderSpec("google-chirp2", "google_speech", {
            "model": "chirp_2", "language_codes": [lang]}),
        # All local ASR via the unified OpenAI /v1/audio/transcriptions client.
        ProviderSpec("qwen3-vllm", "openai_transcription", {
            "base_url": "http://127.0.0.1:8101", "model": "Qwen/Qwen3-ASR-1.7B",
            "language": "vi"}),
        ProviderSpec("funasr-mlt", "openai_transcription", {
            "base_url": "http://127.0.0.1:9102",
            "model": "FunAudioLLM/Fun-ASR-MLT-Nano-2512", "language": "vi"}),
        ProviderSpec("nemotron", "openai_transcription", {
            "base_url": "http://127.0.0.1:9103", "language": "vi-VN"}),
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ASR providers")
    parser.add_argument("--audio", type=str, default="demo_16k.wav")
    parser.add_argument("--language", type=str, default="vi-VN")
    args = parser.parse_args()

    bench = ProviderBenchmark(args.audio, language=args.language)
    providers = default_providers(args.language)
    results = bench.run(providers)
    print_report(results, bench._duration)
