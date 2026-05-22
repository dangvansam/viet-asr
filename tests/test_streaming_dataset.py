import os
import json
import torch
from multitalker_asr.data import get_multitalker_dataloader
from loguru import logger


def test_streaming_dataloader():
    # Create a dummy manifest for testing
    dummy_manifest = "data/dummy_single_speaker.json"
    os.makedirs("data", exist_ok=True)

    # We need a real audio file or a mock.
    # Since I don't want to depend on existing audio for this test,
    # I'll check if any wav exists in the data root or use a dummy path.
    # Note: MultiTalkerMixer will fail if it can't read the file.

    # Let's try to find a wav file in the workspace
    wav_file = None
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".wav"):
                wav_file = os.path.abspath(os.path.join(root, file))
                break
        if wav_file:
            break

    if not wav_file:
        logger.warning(
            "No .wav file found for testing. Creating a dummy sine wave.")
        import numpy as np
        import soundfile as sf
        wav_file = os.path.abspath("data/test_sine.wav")
        sr = 16000
        t = np.linspace(0, 1, sr)
        y = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(wav_file, y, sr)

    with open(dummy_manifest, 'w', encoding='utf-8') as f:
        for i in range(10):
            entry = {
                "audio_filepath": wav_file,
                "text": f"sample text {i}",
                "label": f"speaker_{i % 2}",
                "offset": 0.0,
                "duration": 1.0
            }
            f.write(json.dumps(entry) + '\n')

    logger.info("Initializing streaming dataloader...")
    dataloader = get_multitalker_dataloader(
        manifest_paths=[dummy_manifest],
        batch_size=2,
        max_speakers=2,
        num_workers=0  # Use 0 for easier debugging
    )

    logger.info("Iterating through first batch...")
    for i, batch in enumerate(dataloader):
        # batch = (padded_audio, audio_lens, padded_text, text_lens, spk_mask, bg_mask)
        audio, audio_lens, text, text_lens, spk_mask, bg_mask = batch
        logger.success(f"Batch {i} received!")
        logger.info(f"Audio shape: {audio.shape}")
        logger.info(f"Audio lens: {audio_lens}")
        logger.info(f"Text: {text}")

        assert audio.shape[0] == 2
        assert len(audio_lens) == 2

        if i >= 2:  # Test a few batches
            break

    logger.success("Streaming dataloader test passed!")


if __name__ == "__main__":
    test_streaming_dataloader()
