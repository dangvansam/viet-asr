#!/usr/bin/env python3
"""
Analyze speaker-clustering confidence in a manifest for post-filtering / dynamic
threshold tuning. Groups by source FILE (default) or by CHANNEL — NOT platform.

Plots histograms of speaker_consistency + cluster_margin, reports per-file
num_speakers + low-confidence rate + percentiles, and suggests a threshold.

Usage:
    uv run python scripts/cluster_analysis.py \
        --manifest /home/samdv/DATA/crawl-2026/manifest_all.jsonl [--by file|channel]
"""

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

_SOURCE_RE = re.compile(r"^(.*?)_(?:SPEAKER_\d+|SPK_\d+|SEG_\d+)_")


def group_key(entry, by):
    if by == "channel":
        return entry.get("extra", {}).get("channel") or "unknown"
    sid = entry.get("id") or Path(entry.get("audio_filepath", "")).stem
    m = _SOURCE_RE.match(sid)
    return m.group(1) if m else sid


def pct(values, p):
    if not values:
        return None
    return round(statistics.quantiles(values, n=100)[min(p, 99) - 1], 4) if len(values) > 1 else round(values[0], 4)


def build(entries, by):
    cons = [e.get("extra", {}).get("speaker_consistency") for e in entries]
    cons = [c for c in cons if c is not None]
    marg = [e.get("extra", {}).get("cluster_margin") for e in entries]
    marg = [m for m in marg if m is not None]
    thr = [e.get("extra", {}).get("cluster_threshold") for e in entries]
    thr = [t for t in thr if t is not None]

    groups = defaultdict(list)
    for e in entries:
        groups[group_key(e, by)].append(e)
    nspk_per_group = {g: len({e.get("speaker_id") for e in segs}) for g, segs in groups.items()}

    low = sum(1 for c in cons if c < 0.3)
    return {
        "group_by": by,
        "n_segments": len(entries),
        "n_groups": len(groups),
        "with_consistency": len(cons),
        "speaker_consistency": {
            "mean": round(statistics.mean(cons), 4) if cons else None,
            "p10": pct(cons, 10), "p25": pct(cons, 25), "p50": pct(cons, 50), "p90": pct(cons, 90),
            "low_(<0.3)_count": low,
        },
        "cluster_margin": {
            "mean": round(statistics.mean(marg), 4) if marg else None,
            "p10": pct(marg, 10), "p50": pct(marg, 50), "p90": pct(marg, 90),
        },
        "cluster_threshold_used": dict(Counter(round(t, 2) for t in thr)) if thr else {},
        "num_speakers_per_group": dict(Counter(nspk_per_group.values())),
        "suggested_consistency_floor": pct(cons, 10),
        "cons_values": cons, "marg_values": marg,
    }


def plot(rep, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"(skip plot: {exc})")
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    if rep["cons_values"]:
        ax[0].hist(rep["cons_values"], bins=40, color="#3b82f6")
        ax[0].set_title("speaker_consistency"); ax[0].axvline(0.3, color="r", ls="--")
    if rep["marg_values"]:
        ax[1].hist(rep["marg_values"], bins=40, color="#10b981")
        ax[1].set_title("cluster_margin")
    fig.tight_layout(); fig.savefig(out_png, dpi=90); plt.close(fig)
    print(f"wrote {out_png}")


def main():
    p = argparse.ArgumentParser(description="Speaker-clustering confidence analysis")
    p.add_argument("--manifest", required=True)
    p.add_argument("--by", choices=["file", "channel"], default="file")
    args = p.parse_args()
    path = Path(args.manifest)
    entries = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rep = build(entries, args.by)

    print(f"\n=== CLUSTER CONFIDENCE (by {args.by}) — {rep['n_segments']} seg / {rep['n_groups']} groups ===")
    print(f"consistency: {rep['speaker_consistency']}")
    print(f"margin:      {rep['cluster_margin']}")
    print(f"num_speakers/group: {rep['num_speakers_per_group']}")
    print(f"threshold used: {rep['cluster_threshold_used']}")
    print(f"→ suggested consistency floor (p10): {rep['suggested_consistency_floor']}")

    rep_save = {k: v for k, v in rep.items() if k not in ("cons_values", "marg_values")}
    (path.parent / "cluster_analysis.json").write_text(
        json.dumps(rep_save, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(rep, path.parent / "cluster_consistency_hist.png")


if __name__ == "__main__":
    main()
