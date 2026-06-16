#!/usr/bin/env python3
"""
Compare forced-alignment backends on Vietnamese audio.

Runs each aligner on (audio, given transcript) → per-word timestamps, then reports
cross-aligner agreement (word-span IoU + boundary MAE), sanity (monotonic, coverage,
within-duration), RTF, and agreement with the manifest's existing Fun-ASR alignment
(soft reference). No human ground truth — agreement-based evaluation.

Usage:
    uv run python scripts/benchmark_aligners.py \
        --manifest /home/samdv/DATA/crawl-2026/manifest_all.jsonl \
        --backends funasr_nano_align,funasr_align,mms_fa \
        --limit 20 --device cuda:1 --out aligner_report.json

    # single clip:
    uv run python scripts/benchmark_aligners.py --audio a.wav --text "xin chào" \
        --backends funasr_nano_align,mms_fa
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from multitalker_asr.data.pipeline.align_backends import build_align_backend  # noqa: E402
from multitalker_asr.data.pipeline.segment_utils import interval_iou  # noqa: E402
from multitalker_asr.utils.audio import AudioLoader  # noqa: E402


@dataclass
class ClipResult:
    ok: bool = False
    n_words: int = 0
    score: float = 0.0
    monotonic: bool = True
    within_duration: bool = True
    align_seconds: float = 0.0
    rtf: float = 0.0
    words: List[Tuple[str, float, float]] = field(default_factory=list)
    error: str = ""


def load_clips(args) -> List[Dict]:
    if args.audio:
        return [{"audio_filepath": args.audio, "text": args.text or "", "alignment": None}]
    clips = []
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        text = e.get("text_itn") or e.get("text") or ""
        if e.get("audio_filepath") and text and Path(e["audio_filepath"]).exists():
            clips.append({"audio_filepath": e["audio_filepath"], "text": text,
                          "alignment": e.get("alignment")})
        if len(clips) >= args.limit:
            break
    return clips


def align_one(backend, clip, language, duration) -> ClipResult:
    try:
        t0 = time.perf_counter()
        res = backend.align(clip["audio_filepath"], clip["text"], language)
        dt = time.perf_counter() - t0
    except Exception as exc:
        return ClipResult(ok=False, error=str(exc)[:120])
    words = [(w.text, float(w.start_time), float(w.end_time)) for w in res.words]
    monotonic = all(words[i][1] <= words[i + 1][1] + 1e-3 for i in range(len(words) - 1))
    within = all(0.0 <= s <= duration + 0.5 and e <= duration + 0.5 for _, s, e in words) if words else True
    return ClipResult(
        ok=bool(words), n_words=len(words), score=round(res.score, 4),
        monotonic=monotonic, within_duration=within,
        align_seconds=round(dt, 3), rtf=round(dt / duration, 4) if duration else 0.0,
        words=words,
    )


def pair_iou_mae(a: List[Tuple], b: List[Tuple]) -> Tuple[Optional[float], Optional[float]]:
    """Word-span IoU + boundary MAE over index-matched words."""
    n = min(len(a), len(b))
    if n == 0:
        return None, None
    ious, maes = [], []
    for i in range(n):
        ious.append(interval_iou((a[i][1], a[i][2]), (b[i][1], b[i][2])))
        maes.append((abs(a[i][1] - b[i][1]) + abs(a[i][2] - b[i][2])) / 2.0)
    return sum(ious) / n, sum(maes) / n


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    p = argparse.ArgumentParser(description="Compare forced-alignment backends")
    p.add_argument("--manifest")
    p.add_argument("--audio")
    p.add_argument("--text")
    p.add_argument("--backends", required=True, help="comma list of align backend names")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--language", default="Vietnamese")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--align-kwargs", default="{}",
                   help='JSON per-backend ctor kwargs, e.g. {"funasr_nano_align":{"model":"/path"}}')
    p.add_argument("--out", default=None)
    args = p.parse_args()

    names = [b.strip() for b in args.backends.split(",") if b.strip()]
    align_kwargs = json.loads(args.align_kwargs)
    clips = load_clips(args)
    if not clips:
        print("No clips found.")
        return
    loader = AudioLoader()
    durations = [loader.get_duration(c["audio_filepath"]) for c in clips]
    print(f"Aligning {len(clips)} clip(s) with {names}")

    # results[name] = list of ClipResult (per clip); load each backend once.
    results: Dict[str, List[ClipResult]] = {}
    for name in names:
        try:
            # device goes to load() (uniform across backends), not the constructor
            # — qwen3/nemo_nfa constructors don't take a `device` kwarg.
            backend = build_align_backend(name, **align_kwargs.get(name, {}))
            backend.load(device=args.device)
        except Exception as exc:
            print(f"  backend '{name}' unavailable: {str(exc)[:100]}")
            results[name] = [ClipResult(ok=False, error=str(exc)[:120]) for _ in clips]
            continue
        per = []
        for clip, dur in zip(clips, durations):
            per.append(align_one(backend, clip, args.language, dur))
        results[name] = per
        try:
            backend.unload()
        except Exception:
            pass

    report = build_and_print(names, clips, durations, results)
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


def build_and_print(names, clips, durations, results) -> Dict:
    print("\n" + "=" * 92)
    print(f"FORCED-ALIGNMENT BENCHMARK  ({len(clips)} clips, lang-agnostic agreement)")
    print("=" * 92)
    print(f"{'ALIGNER':<20}{'OK':<6}{'WORDS/clip':<12}{'SCORE':<8}{'MONO%':<8}{'INDUR%':<8}{'RTF':<8}")
    print("-" * 92)
    summary = {}
    for name in names:
        per = results[name]
        ok = [r for r in per if r.ok]
        n_ok = len(ok)
        avg_words = mean([r.n_words for r in ok])
        avg_score = mean([r.score for r in ok])
        mono = mean([1.0 if r.monotonic else 0.0 for r in ok])
        indur = mean([1.0 if r.within_duration else 0.0 for r in ok])
        rtf = mean([r.rtf for r in ok])
        summary[name] = {
            "ok_clips": n_ok, "total_clips": len(per),
            "avg_words": round(avg_words, 1) if avg_words else None,
            "avg_score": round(avg_score, 3) if avg_score else None,
            "monotonic_frac": round(mono, 3) if mono is not None else None,
            "within_duration_frac": round(indur, 3) if indur is not None else None,
            "avg_rtf": round(rtf, 4) if rtf else None,
        }
        f = lambda v, d="-": (d if v is None else v)
        print(f"{name:<20}{f(n_ok):<6}{str(f(summary[name]['avg_words'])):<12}"
              f"{str(f(summary[name]['avg_score'])):<8}{str(f(summary[name]['monotonic_frac'])):<8}"
              f"{str(f(summary[name]['within_duration_frac'])):<8}{str(f(summary[name]['avg_rtf'])):<8}")
    print("-" * 92)

    # Pairwise agreement (mean over clips where both produced words)
    pair = {}
    avail = [n for n in names if any(r.ok for r in results[n])]
    if len(avail) >= 2:
        print("\nPAIRWISE WORD-SPAN IoU  (upper) / BOUNDARY MAE seconds (lower)")
        print("-" * 92)
        print(f"{'':<20}" + "".join(f"{n[:12]:<14}" for n in avail))
        for ni in avail:
            row_iou = f"{ni:<20}"
            for nj in avail:
                ious, maes = [], []
                for ci in range(len(clips)):
                    a, b = results[ni][ci], results[nj][ci]
                    if a.ok and b.ok:
                        iou, mae = pair_iou_mae(a.words, b.words)
                        if iou is not None:
                            ious.append(iou); maes.append(mae)
                miou, mmae = mean(ious), mean(maes)
                pair[f"{ni}|{nj}"] = {"iou": round(miou, 3) if miou is not None else None,
                                      "mae_s": round(mmae, 3) if mmae is not None else None}
                cell = f"{miou:.2f}/{mmae:.2f}" if miou is not None else "-"
                row_iou += f"{cell:<14}"
            print(row_iou)
        print("-" * 92)

    # Consensus reference: per word index, median [start,end] across OK aligners.
    # (The manifest's own `alignment` is a degenerate uniform-60ms-per-token
    # placeholder, not real word timing, so it can't serve as a reference.)
    soft = {}
    import statistics as _st
    print("\nVS CONSENSUS (median word boundary across aligners): IoU / MAE(s)  [lower MAE = closer to consensus]")
    print("-" * 92)
    consensus_per_clip = []
    for ci in range(len(clips)):
        oks = [results[n][ci].words for n in names if results[n][ci].ok and results[n][ci].words]
        if len(oks) < 2:
            consensus_per_clip.append(None)
            continue
        m = min(len(w) for w in oks)
        cons = []
        for i in range(m):
            starts = [w[i][1] for w in oks]
            ends = [w[i][2] for w in oks]
            cons.append((oks[0][i][0], _st.median(starts), _st.median(ends)))
        consensus_per_clip.append(cons)
    for name in names:
        ious, maes = [], []
        for ci in range(len(clips)):
            cons = consensus_per_clip[ci]
            r = results[name][ci]
            if not cons or not r.ok:
                continue
            iou, mae = pair_iou_mae(r.words, cons)
            if iou is not None:
                ious.append(iou); maes.append(mae)
        miou, mmae = mean(ious), mean(maes)
        soft[name] = {"iou": round(miou, 3) if miou is not None else None,
                      "mae_s": round(mmae, 3) if mmae is not None else None}
        print(f"{name:<20}{(f'{miou:.3f}' if miou is not None else '-'):<10}"
              f"{(f'{mmae:.3f}' if mmae is not None else '-')}")
    print("-" * 92)

    print("\nERRORS")
    print("-" * 92)
    for name in names:
        errs = {r.error for r in results[name] if r.error}
        if errs:
            print(f"{name}: {list(errs)[:3]}")
    return {"summary": summary, "pairwise": pair, "vs_manifest": soft, "n_clips": len(clips)}


if __name__ == "__main__":
    main()
