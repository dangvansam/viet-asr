import json
import os
import random
from typing import Any, Dict, List, Optional

import soundfile as sf
from loguru import logger
from tqdm import tqdm

from ..configs import EvalConfig
from ..data.mixers.multitalker import MultiTalkerMixer
from ..utils.formats.audacity import AudacityConverter
from ..utils.formats.rttm import RTTMConverter


class EvalDataSynthesizer:
    def __init__(self, sample_rate: int = 16000):
        self._sample_rate = sample_rate
        self._mixer = MultiTalkerMixer(sr_target=sample_rate)

    def synthesize(self, cfg: EvalConfig) -> List[Dict[str, Any]]:
        if not cfg.source_manifest or not os.path.exists(cfg.source_manifest):
            raise FileNotFoundError(f"Source manifest not found: {cfg.source_manifest}")

        utterances = self._load_utterances(cfg.source_manifest)

        if len(utterances) < cfg.max_speakers:
            raise ValueError(f"Need at least {cfg.max_speakers} utterances, got {len(utterances)}")

        logger.info(f"Loaded {len(utterances)} utterances from {cfg.source_manifest}")

        audio_dir = os.path.join(cfg.eval_data_dir, "audio")
        rttm_dir = os.path.join(cfg.eval_data_dir, "rttm")
        labels_dir = os.path.join(cfg.eval_data_dir, "labels")
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(rttm_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        eval_manifest = []

        for i in tqdm(range(cfg.num_samples), desc="Synthesizing eval samples"):
            sample_id = f"eval_{i:05d}"
            num_spk = random.randint(cfg.min_speakers, cfg.max_speakers)
            selected = random.sample(utterances, num_spk)

            mixed_audio, supervisions, duration = self._mixer.mix(selected, sample_id=sample_id)

            if mixed_audio is None:
                logger.warning(f"Failed to synthesize {sample_id}, skipping")
                continue

            audio_path = os.path.join(audio_dir, f"{sample_id}.wav")
            sf.write(audio_path, mixed_audio, self._sample_rate)

            rttm_path = os.path.join(rttm_dir, f"{sample_id}.rttm")
            rttm_lines = RTTMConverter.from_supervisions(supervisions, sample_id)
            RTTMConverter.write_file(rttm_lines, rttm_path)

            label_path = os.path.join(labels_dir, f"{sample_id}_ref.txt")
            AudacityConverter.from_supervisions(supervisions, label_path)

            eval_manifest.append({
                "audio_filepath": os.path.abspath(audio_path),
                "rttm_filepath": os.path.abspath(rttm_path),
                "num_speakers": num_spk,
                "duration": round(duration, 4),
                "sample_id": sample_id,
            })

        manifest_path = os.path.join(cfg.eval_data_dir, "eval_manifest.json")
        self._write_manifest(eval_manifest, manifest_path)

        logger.info(f"Synthesized {len(eval_manifest)} samples -> {cfg.eval_data_dir}")
        return eval_manifest

    def load_from_dirs(
        self,
        audio_dir: str,
        rttm_dir: str,
    ) -> List[Dict[str, Any]]:
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

                speakers = set()
                with open(os.path.join(rttm_dir, fname), "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 8:
                            speakers.add(parts[7])

                info = sf.info(audio_files[stem])

                eval_manifest.append({
                    "audio_filepath": os.path.abspath(audio_files[stem]),
                    "rttm_filepath": os.path.abspath(os.path.join(rttm_dir, fname)),
                    "num_speakers": len(speakers),
                    "duration": round(info.duration, 4),
                    "sample_id": stem,
                })

        logger.info(f"Loaded {len(eval_manifest)} samples from {audio_dir} + {rttm_dir}")
        return eval_manifest

    def _load_utterances(self, manifest_path: str) -> List[Dict[str, Any]]:
        utterances = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    utterances.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return utterances

    def _write_manifest(
        self,
        manifest: List[Dict[str, Any]],
        output_path: str,
    ) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in manifest:
                f.write(json.dumps(entry) + "\n")
