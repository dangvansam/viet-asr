#!/usr/bin/env python3
"""
Gom các metric chất lượng dataset (1 JSON) để so sánh A/B giữa các vòng tinh chỉnh
pipeline. Đọc manifest JSONL (shard hoặc manifest_all). Bổ sung cho dataset_report.py:
tập trung vào tín hiệu precision cho training — ASR-agreement (ROVER), alignment_score,
char/s, speaker_consistency, attribute coverage.

    uv run python scripts/eval_dataset_quality.py --manifest <manifest.jsonl> [--out eval.json]
    uv run python scripts/eval_dataset_quality.py --diff a/eval.json b/eval.json
"""

import argparse
import glob
import json
import os
import statistics as st
from collections import Counter

_AXES = ("gender", "age", "region", "emotion", "voice_state", "language")


def _pct(num, den):
    return round(100.0 * num / den, 1) if den else 0.0


def _stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    xs.sort()
    return {
        "n": len(xs),
        "mean": round(sum(xs) / len(xs), 4),
        "p10": round(xs[max(0, int(0.10 * len(xs)) - 1)], 4),
        "p50": round(xs[len(xs) // 2], 4),
        "p90": round(xs[min(len(xs) - 1, int(0.90 * len(xs)))], 4),
    }


def _load(manifest):
    paths = [manifest]
    if not os.path.exists(manifest):
        # try shards next to it (write_manifest shards by platform)
        base = manifest.rsplit(".", 1)[0]
        paths = glob.glob(f"{base}*.jsonl") or glob.glob(
            os.path.join(os.path.dirname(manifest) or ".", "*.jsonl"))
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def evaluate(manifest):
    rows = _load(manifest)
    n = len(rows)
    durs = [float(r.get("duration", 0.0)) for r in rows]
    align = [r.get("alignment_score") for r in rows if r.get("alignment_score") is not None]
    asr_agree, snr, consist, subsim, charps = [], [], [], [], []
    text_ok = align_hi = 0
    attr_real = Counter()
    for r in rows:
        e = r.get("extra", {}) or {}
        txt = (r.get("text") or "").strip()
        if txt:
            text_ok += 1
        d = float(r.get("duration", 0.0) or 0.0)
        if txt and d > 0:
            charps.append(len(txt.replace(" ", "")) / d)
        if r.get("alignment_score") is not None and float(r["alignment_score"]) >= 0.85:
            align_hi += 1
        asr_c = r.get("asr_confidence", e.get("asr_agreement"))
        if isinstance(asr_c, (int, float)):
            asr_agree.append(float(asr_c))
        for key, dst in (("snr_db", snr), ("speaker_consistency", consist),
                         ("subtitle_similarity", subsim)):
            v = e.get(key)
            if isinstance(v, (int, float)):
                dst.append(float(v))
        conf = r.get("attribute_confidence", {}) or {}
        for ax in _AXES:
            if float(conf.get(ax, 0.0) or 0.0) > 0.0:
                attr_real[ax] += 1

    return {
        "manifest": manifest,
        "yield": {
            "n_segments": n,
            "total_hours": round(sum(durs) / 3600.0, 3),
            "text_coverage_pct": _pct(text_ok, n),
            "duration_s": _stats(durs),
        },
        "transcript": {
            "asr_agreement": _stats(asr_agree),
            "subtitle_similarity": _stats(subsim),
            "char_per_sec": _stats(charps),
        },
        "alignment": {
            "alignment_score": _stats(align),
            "pct_score_ge_0.85": _pct(align_hi, n),
        },
        "speaker": {
            "consistency": _stats(consist),
            "pct_low_lt_0.3": _pct(sum(1 for c in consist if c < 0.3), len(consist)),
        },
        "audio": {"snr_db": _stats(snr)},
        "attribute_coverage_pct": {ax: _pct(attr_real[ax], n) for ax in _AXES},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--out")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.diff:
        a, b = (json.load(open(p)) for p in args.diff)
        print(f"A = {args.diff[0]}\nB = {args.diff[1]}\n")
        for sec in ("yield", "transcript", "alignment", "speaker", "audio"):
            print(f"[{sec}]")
            print("  A:", json.dumps(a.get(sec), ensure_ascii=False))
            print("  B:", json.dumps(b.get(sec), ensure_ascii=False))
        print("[attribute_coverage_pct]")
        print("  A:", a.get("attribute_coverage_pct"))
        print("  B:", b.get("attribute_coverage_pct"))
        return

    report = evaluate(args.manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    out = args.out or os.path.join(os.path.dirname(args.manifest) or ".", "eval_dataset_quality.json")
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
