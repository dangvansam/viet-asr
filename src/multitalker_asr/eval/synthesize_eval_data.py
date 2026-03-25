import os
import json
import random

import numpy as np
import soundfile as sf
from loguru import logger
from tqdm import tqdm

from multitalker_asr.data.mixer import MultiTalkerMixer


def supervisions_to_rttm(supervisions, sample_id):
    """Convert lhotse SupervisionSegments to RTTM lines."""
    lines = []
    for seg in supervisions:
        lines.append(
            f"SPEAKER {sample_id} 1 {seg.start:.4f} {seg.duration:.4f} "
            f"<NA> <NA> {seg.speaker} <NA> <NA>"
        )
    return lines


def supervisions_to_audacity_labels(supervisions):
    """Convert lhotse SupervisionSegments to Audacity label format.

    Audacity labels: start_seconds\\tend_seconds\\tlabel
    """
    lines = []
    for seg in supervisions:
        end = seg.start + seg.duration
        lines.append(f"{seg.start:.6f}\t{end:.6f}\t{seg.speaker}")
    return lines


def synthesize_eval_set(cfg):
    """Synthesize a multi-speaker evaluation set from single-speaker manifests.

    Args:
        cfg: EvalConfig with source_manifest, eval_data_dir, num_samples,
             min_speakers, max_speakers fields.

    Returns:
        List of dicts with keys: audio_filepath, rttm_filepath, num_speakers, duration.
    """
    if not cfg.source_manifest or not os.path.exists(cfg.source_manifest):
        raise FileNotFoundError(
            f"Source manifest not found: {cfg.source_manifest}"
        )

    # Load all single-speaker utterances
    utterances = []
    with open(cfg.source_manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                utterances.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if len(utterances) < cfg.max_speakers:
        raise ValueError(
            f"Need at least {cfg.max_speakers} utterances, got {len(utterances)}"
        )

    logger.info(
        f"Loaded {len(utterances)} utterances from {cfg.source_manifest}"
    )

    # Create output directories
    audio_dir = os.path.join(cfg.eval_data_dir, "audio")
    rttm_dir = os.path.join(cfg.eval_data_dir, "rttm")
    labels_dir = os.path.join(cfg.eval_data_dir, "labels")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(rttm_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    mixer = MultiTalkerMixer(sr_target=16000)
    eval_manifest = []

    for i in tqdm(range(cfg.num_samples), desc="Synthesizing eval samples"):
        sample_id = f"eval_{i:05d}"
        num_spk = random.randint(cfg.min_speakers, cfg.max_speakers)
        selected = random.sample(utterances, num_spk)

        mixed_audio, supervisions, duration = mixer.mix_utterances(
            selected, sample_id=sample_id
        )

        if mixed_audio is None:
            logger.warning(f"Failed to synthesize {sample_id}, skipping")
            continue

        # Save audio
        audio_path = os.path.join(audio_dir, f"{sample_id}.wav")
        sf.write(audio_path, mixed_audio, 16000)

        # Save ground truth RTTM
        rttm_path = os.path.join(rttm_dir, f"{sample_id}.rttm")
        rttm_lines = supervisions_to_rttm(supervisions, sample_id)
        with open(rttm_path, "w") as f:
            f.write("\n".join(rttm_lines) + "\n")

        # Save ground truth Audacity labels
        label_path = os.path.join(labels_dir, f"{sample_id}_ref.txt")
        label_lines = supervisions_to_audacity_labels(supervisions)
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines) + "\n")

        eval_manifest.append({
            "audio_filepath": os.path.abspath(audio_path),
            "rttm_filepath": os.path.abspath(rttm_path),
            "num_speakers": num_spk,
            "duration": round(duration, 4),
            "sample_id": sample_id,
        })

    # Save manifest for later reuse
    manifest_path = os.path.join(cfg.eval_data_dir, "eval_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in eval_manifest:
            f.write(json.dumps(entry) + "\n")

    logger.info(
        f"Synthesized {len(eval_manifest)} samples -> {cfg.eval_data_dir}"
    )
    return eval_manifest


def load_eval_manifest_from_dirs(audio_dir, rttm_dir):
    """Build an eval manifest from pre-existing audio and RTTM directories.

    Matches files by stem name (e.g., sample_00000.wav <-> sample_00000.rttm).
    """
    audio_files = {}
    for fname in sorted(os.listdir(audio_dir)):
        if fname.endswith(".wav"):
            stem = os.path.splitext(fname)[0]
            audio_files[stem] = os.path.join(audio_dir, fname)

    eval_manifest = []
    for fname in sorted(os.listdir(rttm_dir)):
        if fname.endswith(".rttm"):
            stem = os.path.splitext(fname)[0]
            if stem not in audio_files:
                logger.warning(f"No audio found for RTTM {fname}, skipping")
                continue

            # Count speakers from RTTM
            speakers = set()
            with open(os.path.join(rttm_dir, fname), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        speakers.add(parts[7])

            # Get duration from audio
            info = sf.info(audio_files[stem])

            eval_manifest.append({
                "audio_filepath": os.path.abspath(audio_files[stem]),
                "rttm_filepath": os.path.abspath(os.path.join(rttm_dir, fname)),
                "num_speakers": len(speakers),
                "duration": round(info.duration, 4),
                "sample_id": stem,
            })

    logger.info(
        f"Loaded {len(eval_manifest)} samples from {audio_dir} + {rttm_dir}"
    )
    return eval_manifest
