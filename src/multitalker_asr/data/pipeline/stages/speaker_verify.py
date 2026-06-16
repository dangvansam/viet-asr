"""
SpeakerVerifyStage: re-cluster speaker embeddings (speaker-recognition /embed
service, ECAPA-TDNN) per source file to VERIFY/CORRECT diarization labels.

- Relabel every segment to its voice cluster (SPK_<k>) — fixes diarization
  label errors.
- Drop a segment if it does not fit its assigned cluster (low leave-one-out
  consistency) AND it is not flagged as overlap. Lone-speaker singletons and
  overlap segments are kept.
- Sets num_speakers (clusters per file) and extra.speaker_consistency.
Runs after vad_diarize (any backend: pyannote/sortformer give labels to verify;
vad_sv gives placeholder labels to assign).
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np
from loguru import logger

from ..base_stage import BaseStage
from ..checkpoint import PipelineCheckpoint
from ..config import PipelineConfig
from ..parallel import parallel_map
from ..speaker import (
    SpeakerEmbedder,
    centroids,
    cluster_embeddings,
    cluster_margin,
    consistency_scores,
    cosine,
    estimate_threshold,
)


class SpeakerVerifyStage(BaseStage):
    name = "speaker_verify"

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        cfg = config.speaker_verify
        if not cfg.enabled:
            return records

        embedder = SpeakerEmbedder(cfg.url, cfg.timeout)
        if not embedder.health():
            logger.warning(f"Speaker service unavailable at {cfg.url}, skipping speaker_verify")
            return records

        workers = getattr(config, "concurrency", 1) or 1
        emb_list = parallel_map(lambda r: embedder.embed(r["audio_filepath"]), records, workers)
        emb_by_id = {r["id"]: e for r, e in zip(records, emb_list)}

        groups: Dict[str, List[Dict]] = defaultdict(list)
        for r in records:
            groups[r.get("source_audio") or r["id"]].append(r)

        kept: List[Dict] = []
        stats = {"relabeled": 0, "dropped": 0, "embed_fail": 0, "files": len(groups)}

        for _, segs in groups.items():
            kept.extend(self._verify_group(segs, emb_by_id, cfg, checkpoint, stats))

        logger.info(
            f"SpeakerVerify: {len(kept)}/{len(records)} kept | "
            f"relabeled={stats['relabeled']} dropped={stats['dropped']} "
            f"embed_fail={stats['embed_fail']} files={stats['files']}"
        )
        checkpoint.save_state()
        return kept

    def _verify_group(self, segs, emb_by_id, cfg, checkpoint, stats) -> List[Dict]:
        embs: List[np.ndarray] = []
        valid: List[Dict] = []
        passthrough: List[Dict] = []
        for s in segs:
            e = emb_by_id.get(s["id"])
            if e is None:
                stats["embed_fail"] += 1
                passthrough.append(dict(s))
            else:
                embs.append(e)
                valid.append(s)

        if len(valid) < cfg.min_cluster_size:
            for s in valid:
                s = dict(s)
                s["num_speakers"] = len({x["speaker_id"] for x in valid}) or 1
                checkpoint.mark_processed(s["id"], self.name)
                passthrough.append(s)
            return passthrough

        matrix = np.stack(embs)
        threshold = self._resolve_threshold(cfg, matrix)
        labels = cluster_embeddings(matrix, threshold)
        cents = centroids(matrix, labels)
        scores = consistency_scores(matrix, labels)
        sizes = {int(l): int((labels == l).sum()) for l in set(labels.tolist())}
        num_spk = len(sizes)

        out: List[Dict] = list(passthrough)
        for s, lab, score, emb in zip(valid, labels, scores, embs):
            s = dict(s)
            s["num_speakers"] = num_spk
            centroid_sim = cosine(emb, cents[int(lab)])
            extra = dict(s.get("extra") or {})
            extra["speaker_consistency"] = round(float(score), 4)
            extra["cluster_centroid_sim"] = round(float(centroid_sim), 4)
            extra["cluster_margin"] = cluster_margin(emb, int(lab), cents)
            extra["cluster_threshold"] = round(float(threshold), 4)
            extra["cluster_size"] = sizes[int(lab)]
            is_overlap = bool(extra.get("is_overlap", False))

            new_spk = f"SPK_{int(lab)}"
            if new_spk != s.get("speaker_id"):
                stats["relabeled"] += 1
                extra["relabeled_from"] = s.get("speaker_id")
                s["speaker_id"] = new_spk

            # Drop a clearly-inconsistent member (not overlap, not a lone singleton).
            if (
                cfg.drop_ambiguous
                and not is_overlap
                and score < cfg.ambiguous_max
                and centroid_sim < cfg.relabel_min
            ):
                stats["dropped"] += 1
                continue

            s["extra"] = extra
            checkpoint.mark_processed(s["id"], self.name)
            out.append(s)
        return out

    def _resolve_threshold(self, cfg, matrix) -> float:
        """Per-file threshold: 'auto' estimates from this file's similarity
        distribution; otherwise the configured static value."""
        thr = cfg.cluster_threshold
        if isinstance(thr, str) and thr.lower() == "auto":
            return estimate_threshold(matrix)
        return float(thr)
