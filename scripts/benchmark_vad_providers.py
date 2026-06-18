import argparse
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
from loguru import logger

from multitalker_asr.data.pipeline.vad_backends import build_vad_backend


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
    detect_seconds: float = 0.0
    rtf: float = 0.0
    speech_ratio: float = 0.0
    num_segments: int = 0
    speech_seconds: float = 0.0
    avg_seg_s: float = 0.0
    median_seg_s: float = 0.0
    frames: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    error: str = ""


class VADBenchmark:
    def __init__(self, audio_path: str, frame_hop_s: float = 0.02):
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        self._audio = audio
        self._sample_rate = sample_rate
        self._duration = len(audio) / sample_rate
        self._frame_hop_s = frame_hop_s
        self._n_frames = max(1, math.ceil(self._duration / frame_hop_s))
        logger.info(
            f"Loaded {audio_path}: sr={sample_rate} dur={self._duration:.2f}s "
            f"frames={self._n_frames}@{frame_hop_s}s"
        )

    def run(self, providers: List[ProviderSpec]) -> List[ProviderResult]:
        return [self._run_one(spec) for spec in providers]

    def _run_one(self, spec: ProviderSpec) -> ProviderResult:
        logger.info(f"--- {spec.label} ({spec.backend}) ---")
        try:
            backend = build_vad_backend(spec.backend, **spec.kwargs)
            load_start = time.perf_counter()
            backend.load()
            load_seconds = time.perf_counter() - load_start

            det_start = time.perf_counter()
            result = backend.detect(self._audio, self._sample_rate)
            detect_seconds = time.perf_counter() - det_start
            backend.unload()

            durations = [seg.duration for seg in result.segments]
            speech_seconds = sum(durations)
            avg_seg = float(np.mean(durations)) if durations else 0.0
            median_seg = float(np.median(durations)) if durations else 0.0
            rtf = detect_seconds / self._duration if self._duration else 0.0
            logger.success(
                f"{spec.label}: {detect_seconds:.2f}s rtf={rtf:.3f} "
                f"ratio={result.speech_ratio:.3f} segs={len(result.segments)} "
                f"avg_seg={avg_seg:.2f}s"
            )
            return ProviderResult(
                label=spec.label,
                backend=spec.backend,
                ok=True,
                load_seconds=load_seconds,
                detect_seconds=detect_seconds,
                rtf=rtf,
                speech_ratio=result.speech_ratio,
                num_segments=len(result.segments),
                speech_seconds=speech_seconds,
                avg_seg_s=avg_seg,
                median_seg_s=median_seg,
                frames=self._rasterize(result.segments),
            )
        except Exception as exc:
            logger.warning(f"{spec.label} failed: {exc}")
            return ProviderResult(
                label=spec.label, backend=spec.backend, ok=False, error=str(exc)
            )

    def _rasterize(self, segments) -> np.ndarray:
        frames = np.zeros(self._n_frames, dtype=np.int8)
        for seg in segments:
            lo = max(0, int(seg.start / self._frame_hop_s))
            hi = min(self._n_frames, int(math.ceil(seg.end / self._frame_hop_s)))
            if hi > lo:
                frames[lo:hi] = 1
        return frames


def frame_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 1.0


def print_ground_truth(results: List[ProviderResult], gt: np.ndarray) -> None:
    print("\nACCURACY VS GROUND TRUTH (frame-level)")
    print("-" * 118)
    print(f"{'PROVIDER':<22}{'IoU':<10}{'F1':<10}")
    for r in results:
        if not (r.ok and r.frames.size == gt.size):
            continue
        inter = int(np.logical_and(r.frames, gt).sum())
        union = int(np.logical_or(r.frames, gt).sum())
        iou = inter / union if union else 1.0
        denom = int(r.frames.sum()) + int(gt.sum())
        f1 = 2 * inter / denom if denom else 1.0
        print(f"{r.label:<22}{iou:<10.3f}{f1:<10.3f}")
    print("-" * 118)


def print_report(results: List[ProviderResult], duration: float) -> None:
    print("\n" + "=" * 96)
    print(f"VAD PROVIDER BENCHMARK  (audio duration: {duration:.2f}s)")
    print("=" * 96)
    header = (
        f"{'PROVIDER':<22}{'STATUS':<9}{'LOAD(s)':<9}{'DETECT(s)':<11}"
        f"{'RTF':<8}{'RATIO':<8}{'SEGS':<7}{'SPEECH(s)':<10}{'AVG_SEG(s)':<11}{'MED_SEG(s)':<11}"
    )
    print(header)
    print("-" * 118)
    for r in results:
        status = "OK" if r.ok else "FAIL"
        load = f"{r.load_seconds:.2f}" if r.ok else "-"
        det = f"{r.detect_seconds:.2f}" if r.ok else "-"
        rtf = f"{r.rtf:.3f}" if r.ok else "-"
        ratio = f"{r.speech_ratio:.3f}" if r.ok else "-"
        segs = str(r.num_segments) if r.ok else "-"
        speech = f"{r.speech_seconds:.2f}" if r.ok else "-"
        avg_seg = f"{r.avg_seg_s:.2f}" if r.ok else "-"
        med_seg = f"{r.median_seg_s:.2f}" if r.ok else "-"
        print(
            f"{r.label:<22}{status:<9}{load:<9}{det:<11}{rtf:<8}{ratio:<8}{segs:<7}"
            f"{speech:<10}{avg_seg:<11}{med_seg:<11}"
        )
    print("-" * 118)

    ok = [r for r in results if r.ok and r.frames.size]
    if len(ok) >= 2:
        print("\nPAIRWISE FRAME-LEVEL IoU AGREEMENT")
        print("-" * 96)
        labels = [r.label for r in ok]
        print(f"{'':<22}" + "".join(f"{lab[:10]:<12}" for lab in labels))
        for ri in ok:
            row = f"{ri.label:<22}"
            for rj in ok:
                row += f"{frame_iou(ri.frames, rj.frames):<12.3f}"
            print(row)
        print("-" * 96)

        stack = np.stack([r.frames for r in ok], axis=0)
        majority = (stack.sum(axis=0) >= math.ceil(len(ok) / 2)).astype(np.int8)
        print("\nDISAGREEMENT VS MAJORITY (fraction of frames differing)")
        print("-" * 96)
        for r in ok:
            disagree = float(np.mean(r.frames != majority))
            print(f"{r.label:<22}{disagree:.3f}")
        print("-" * 96)

    print("\nERRORS")
    print("-" * 96)
    for r in results:
        if not r.ok:
            print(f"[{r.label}] ERROR: {r.error}")
    print("=" * 96 + "\n")


def default_providers(hf_token: str) -> List[ProviderSpec]:
    return [
        ProviderSpec("silero", "silero", {}),
        ProviderSpec("pyannote-seg3", "pyannote_seg", {"hf_token": hf_token or None}),
        ProviderSpec("ten-vad", "ten", {}),
        ProviderSpec("consensus-majority", "consensus", {
            "strategy": "majority",
            "hf_token": hf_token or None,
            "providers": [{"name": "silero"}, {"name": "pyannote_seg"}, {"name": "ten"}],
        }),
    ]


def dynamic_pairs(hf_token: str, schedule=None) -> List[ProviderSpec]:
    """Pair each base backend with its dynamic-VAD-wrapped counterpart (off vs on)."""
    bases = [
        ("silero", {}),
        ("pyannote_seg", {"hf_token": hf_token or None}),
        ("ten", {}),
        ("fsmn", {}),
    ]
    specs: List[ProviderSpec] = []
    for name, kwargs in bases:
        specs.append(ProviderSpec(name, name, dict(kwargs)))
        dyn_kwargs = {"provider": name, "provider_kwargs": dict(kwargs)}
        if schedule:
            dyn_kwargs["silence_schedule"] = schedule
        if name in ("pyannote_seg",) and hf_token:
            dyn_kwargs["hf_token"] = hf_token
        specs.append(ProviderSpec(f"{name}+dyn", "dynamic", dyn_kwargs))
    return specs


def load_ground_truth(path: str, n_frames: int, frame_hop_s: float) -> np.ndarray:
    """Rasterize a reference VAD label file into a per-frame int8 speech vector.

    Accepts RTTM (SPEAKER lines: field[3]=start, field[4]=dur) or JSON list of
    {"start","end"} objects.
    """
    import json
    import os

    spans: List[tuple] = []
    if path.lower().endswith(".rttm"):
        with open(path) as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 5 and parts[0] == "SPEAKER":
                    start = float(parts[3])
                    spans.append((start, start + float(parts[4])))
    else:
        with open(path) as handle:
            for obj in json.load(handle):
                spans.append((float(obj["start"]), float(obj["end"])))

    frames = np.zeros(n_frames, dtype=np.int8)
    for start, end in spans:
        lo = max(0, int(start / frame_hop_s))
        hi = min(n_frames, int(math.ceil(end / frame_hop_s)))
        if hi > lo:
            frames[lo:hi] = 1
    logger.info(f"Loaded ground truth {os.path.basename(path)}: {len(spans)} spans")
    return frames


if __name__ == "__main__":
    import os

    import json

    parser = argparse.ArgumentParser(description="Benchmark + compare VAD providers")
    parser.add_argument("--audio", type=str, default="demo_16k.wav")
    parser.add_argument("--frame-hop", type=float, default=0.02)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["providers", "dynamic"],
        default="providers",
        help="'providers' = compare backends; 'dynamic' = each backend dynamic off vs on",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="",
        help="JSON silence schedule for dynamic mode, e.g. '[[5000,1500],[1e9,300]]'",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="",
        help="RTTM or JSON reference VAD labels for frame-level IoU/F1",
    )
    parser.add_argument(
        "--providers",
        type=str,
        default="",
        help="comma-separated subset of provider labels to run (default: all)",
    )
    args = parser.parse_args()

    bench = VADBenchmark(args.audio, frame_hop_s=args.frame_hop)
    hf_token = os.environ.get("HF_TOKEN", "")
    if args.mode == "dynamic":
        schedule = json.loads(args.schedule) if args.schedule else None
        providers = dynamic_pairs(hf_token, schedule)
    else:
        providers = default_providers(hf_token)
    if args.providers:
        wanted = {p.strip() for p in args.providers.split(",")}
        providers = [p for p in providers if p.label in wanted]
    results = bench.run(providers)
    print_report(results, bench._duration)

    if args.ground_truth:
        gt = load_ground_truth(args.ground_truth, bench._n_frames, args.frame_hop)
        print_ground_truth(results, gt)
