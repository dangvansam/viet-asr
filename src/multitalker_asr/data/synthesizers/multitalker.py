import json
import os
import random
from typing import Any, Dict, List, Optional

import soundfile as sf
from joblib import Parallel, delayed
from lhotse import AudioSource, CutSet, MonoCut, Recording
from loguru import logger
from tqdm import tqdm

from ..mixers.multitalker import MultiTalkerMixer


class MultitalkerSynthesizer:
    def __init__(
        self,
        sample_rate: int = 16000,
        n_jobs: int = 32,
    ):
        self._sample_rate = sample_rate
        self._n_jobs = n_jobs
        self._mixer = MultiTalkerMixer(sr_target=sample_rate)

    def synthesize(
        self,
        input_manifests: List[str],
        output_dir: str,
        output_manifest: str,
        num_samples: int = 1000,
        max_speakers: int = 2,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        utterances = self._load_utterances(input_manifests)

        if not utterances:
            logger.error("No utterances found for synthesis.")
            return

        if num_samples <= 0:
            num_samples = len(utterances)

        logger.info(f"Synthesizing {num_samples} mixed samples...")

        job_args = self._prepare_jobs(utterances, num_samples, max_speakers)
        results = Parallel(n_jobs=self._n_jobs, pre_dispatch="2.5*n_jobs")(
            delayed(self._process_sample)(i, utts, output_dir)
            for i, utts in tqdm(job_args)
        )

        cuts = [res for res in results if res is not None]
        CutSet.from_cuts(cuts).to_file(output_manifest)
        logger.success(f"Synthesized Lhotse CutSet saved to {output_manifest}")

    def _load_utterances(self, manifest_paths: List[str]) -> List[Dict[str, Any]]:
        utterances = []
        for path in manifest_paths:
            if not os.path.exists(path):
                logger.warning(f"Manifest {path} not found.")
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        utterances.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        return utterances

    def _prepare_jobs(
        self,
        utterances: List[Dict[str, Any]],
        num_samples: int,
        max_speakers: int,
    ) -> List[tuple]:
        job_args = []
        for i in range(num_samples):
            num_spk = random.randint(2, max_speakers)
            utts = random.sample(utterances, num_spk)
            job_args.append((i, utts))
        return job_args

    def _process_sample(
        self,
        index: int,
        utterances: List[Dict[str, Any]],
        output_dir: str,
    ) -> Optional[MonoCut]:
        mix_id = f"mixed_{index:05d}"
        mixed_audio, supervisions, total_duration = self._mixer.mix(
            utterances, sample_id=mix_id
        )

        if mixed_audio is None:
            return None

        out_audio_path = os.path.join(output_dir, f"{mix_id}.wav")
        sf.write(out_audio_path, mixed_audio, self._sample_rate)

        recording = Recording(
            id=mix_id,
            sources=[
                AudioSource(
                    type="file",
                    channels=[0],
                    source=os.path.abspath(out_audio_path),
                )
            ],
            sampling_rate=self._sample_rate,
            num_samples=len(mixed_audio),
            duration=total_duration,
            channel_ids=[0],
        )

        return MonoCut(
            id=mix_id,
            start=0.0,
            duration=total_duration,
            channel=0,
            recording=recording,
            supervisions=supervisions,
        )
