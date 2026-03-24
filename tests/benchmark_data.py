import os
import time
import torch
from multitalker_asr.data.streaming import get_multitalker_dataloader
from loguru import logger
import json


def benchmark_dataloader(num_workers=4, batch_size=32, num_batches=100):
    dummy_manifest = "data/dummy_single_speaker.json"
    if not os.path.exists(dummy_manifest):
        logger.error(
            "Dummy manifest not found. Run test_streaming_dataset.py first.")
        return

    logger.info(
        f"Benchmarking Dataloader (workers: {num_workers}, batch_size: {batch_size}, batches: {num_batches})...")

    dataloader = get_multitalker_dataloader(
        manifest_paths=[dummy_manifest],
        batch_size=batch_size,
        max_speakers=2,
        num_workers=num_workers
    )

    start_time = time.time()
    total_samples = 0

    for i, batch in enumerate(dataloader):
        total_samples += batch['audio_signal'].shape[0]
        if i + 1 >= num_batches:
            break

    end_time = time.time()
    total_time = end_time - start_time
    samples_per_sec = total_samples / total_time
    batches_per_sec = (i + 1) / total_time

    logger.success(f"Benchmarking Complete!")
    logger.info(f"Total Time: {total_time:.2f}s")
    logger.info(f"Throughput: {samples_per_sec:.2f} samples/s")
    logger.info(f"Throughput: {batches_per_sec:.2f} batches/s")

    return samples_per_sec


if __name__ == "__main__":
    # Test with different worker counts
    results = {}
    for nw in [0, 4, 8, 16]:
        results[nw] = benchmark_dataloader(num_workers=nw, num_batches=20)

    logger.info("Summary of Throughput (samples/s):")
    for nw, speed in results.items():
        logger.info(f"  Workers {nw}: {speed:.2f}")
