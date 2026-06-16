#!/usr/bin/env python3
"""
Diversity / completeness report for a training manifest. Prints distribution
across all attribute axes + segment_type + speaker stats, flags imbalance and
missing tags, and writes dataset_report.json next to the manifest.

Usage:
    uv run python scripts/dataset_report.py --manifest /home/samdv/DATA/crawl-2026/manifest_all.jsonl
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

_AXES = ("language", "emotion", "gender", "age", "region", "voice_state")
_DEFAULTS = {"emotion": "neutral", "gender": "unknown", "age": "unknown",
             "region": "unknown", "voice_state": "sober", "language": "vi"}


def build_report(entries):
    n = len(entries)
    total_h = round(sum(float(e.get("duration", 0)) for e in entries) / 3600.0, 3)
    seg_type = Counter(e.get("segment_type", "single") for e in entries)
    nspk = Counter(int(e.get("num_speakers", 1)) for e in entries)

    dist = {ax: dict(Counter(e.get(ax) for e in entries)) for ax in _AXES}

    # "real" = attribute came from a tag/model (confidence > 0), not a fallback default.
    # Falls back to value-not-default for old manifests without per-axis confidence.
    def is_real(e, ax):
        conf = (e.get("attribute_confidence") or {}).get(ax)
        if conf is not None:
            return conf > 0
        return bool(e.get(ax)) and e.get(ax) != _DEFAULTS.get(ax)

    real = {ax: sum(1 for e in entries if is_real(e, ax)) for ax in _AXES}

    has_raw = sum(1 for e in entries if e.get("text_raw"))
    has_itn = sum(1 for e in entries if e.get("text_itn"))
    overlap_with_spk = sum(1 for e in entries
                           if e.get("segment_type") == "overlap"
                           and e.get("extra", {}).get("multitalker_segments"))

    snr = [e.get("extra", {}).get("snr_db") for e in entries]
    snr = [s for s in snr if s is not None]
    cons = [e.get("extra", {}).get("speaker_consistency") for e in entries]
    cons = [c for c in cons if c is not None]
    quality = {
        "snr_db_mean": round(statistics.mean(snr), 2) if snr else None,
        "snr_db_min": round(min(snr), 2) if snr else None,
        "consistency_mean": round(statistics.mean(cons), 4) if cons else None,
        "consistency_low_(<0.3)": sum(1 for c in cons if c < 0.3),
        "with_snr": len(snr),
        "with_consistency": len(cons),
    }

    return {
        "n_segments": n,
        "total_hours": total_h,
        "segment_type": dict(seg_type),
        "num_speakers": dict(sorted(nspk.items())),
        "text_raw_present": has_raw,
        "text_itn_present": has_itn,
        "overlap_with_perspeaker_text": overlap_with_spk,
        "attribute_distribution": dist,
        "non_default_counts": real,
        "quality": quality,
    }


def print_report(rep):
    print(f"\n=== DATASET REPORT: {rep['n_segments']} segments / {rep['total_hours']} h ===")
    print(f"segment_type: {rep['segment_type']}")
    print(f"num_speakers: {rep['num_speakers']}")
    print(f"text_raw present: {rep['text_raw_present']}/{rep['n_segments']} | "
          f"text_itn present: {rep['text_itn_present']}/{rep['n_segments']} | "
          f"overlap w/ per-speaker text: {rep['overlap_with_perspeaker_text']}")
    q = rep["quality"]
    print(f"quality: SNR mean={q['snr_db_mean']} min={q['snr_db_min']} ({q['with_snr']} seg) | "
          f"consistency mean={q['consistency_mean']} low<0.3={q['consistency_low_(<0.3)']} "
          f"({q['with_consistency']} seg)")
    print("\n-- attribute distribution (non-default coverage) --")
    for ax, d in rep["attribute_distribution"].items():
        real = rep["non_default_counts"][ax]
        pct = round(100 * real / rep["n_segments"], 1) if rep["n_segments"] else 0
        top = dict(sorted(d.items(), key=lambda kv: -kv[1])[:6])
        print(f"  {ax:11} real={real} ({pct}%)  {top}")
    print("\n-- warnings --")
    warned = False
    for ax, real in rep["non_default_counts"].items():
        pct = 100 * real / rep["n_segments"] if rep["n_segments"] else 0
        if pct < 30:
            print(f"  ⚠ {ax}: chỉ {pct:.0f}% segment có giá trị thật (còn lại default)")
            warned = True
    if rep["segment_type"].get("overlap", 0) == 0:
        print("  ⚠ chưa có segment overlap (bật multitalker_reprocess)")
        warned = True
    if not warned:
        print("  (không có cảnh báo)")


def main():
    p = argparse.ArgumentParser(description="Dataset diversity/completeness report")
    p.add_argument("--manifest", required=True)
    args = p.parse_args()
    path = Path(args.manifest)
    entries = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rep = build_report(entries)
    print_report(rep)
    out = path.parent / "dataset_report.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
