"""Audio decoding for the transcription endpoints.

WAV (16-bit PCM) is decoded natively with zero dependencies. Any other
container/codec (mp3, m4a, flac, ogg, webm, ...) is decoded via ffmpeg if it
is on PATH. The VietASR core resamples internally, so the native sample rate
is preserved for WAV; ffmpeg output is normalized to 16 kHz mono.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Tuple

import numpy as np

TARGET_RATE = 16000


class AudioDecodeError(ValueError):
    """Raised when an upload cannot be decoded to PCM."""


def _decode_wav(data: bytes) -> Tuple[np.ndarray, int]:
    if len(data) < 44 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioDecodeError("not a RIFF/WAVE file")

    offset = 12
    audio_format = 1
    channels = 1
    sample_rate = TARGET_RATE
    bits = 16

    while offset + 8 <= len(data):
        chunk_id = data[offset:offset + 4]
        size = int.from_bytes(data[offset + 4:offset + 8], "little")
        body = data[offset + 8:offset + 8 + size]
        if chunk_id == b"fmt " and len(body) >= 16:
            audio_format = int.from_bytes(body[0:2], "little")
            channels = max(1, int.from_bytes(body[2:4], "little"))
            sample_rate = int.from_bytes(body[4:8], "little")
            bits = int.from_bytes(body[14:16], "little")
        elif chunk_id == b"data":
            if audio_format != 1 or bits != 16:
                raise AudioDecodeError("WAV must be 16-bit PCM (use ffmpeg for other codecs)")
            samples = np.frombuffer(body, dtype="<i2")
            if channels > 1:
                usable = (samples.size // channels) * channels
                samples = (samples[:usable].reshape(-1, channels)
                           .mean(axis=1).astype(np.int16))
            return np.ascontiguousarray(samples), sample_rate
        offset += 8 + size + (size & 1)

    raise AudioDecodeError("WAV has no data chunk")


def _decode_ffmpeg(data: bytes) -> Tuple[np.ndarray, int]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AudioDecodeError(
            "unsupported audio format: install ffmpeg or upload a 16-bit PCM WAV")
    proc = subprocess.run(
        [ffmpeg, "-nostdin", "-loglevel", "error", "-i", "pipe:0",
         "-f", "s16le", "-ac", "1", "-ar", str(TARGET_RATE), "pipe:1"],
        input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise AudioDecodeError(
            "ffmpeg could not decode the audio" + (f": {detail[-1]}" if detail else ""))
    return np.frombuffer(proc.stdout, dtype="<i2"), TARGET_RATE


def decode_audio(data: bytes) -> Tuple[np.ndarray, int]:
    """Decode an uploaded audio file to (int16 mono PCM, sample_rate)."""
    if not data:
        raise AudioDecodeError("empty upload")
    try:
        return _decode_wav(data)
    except AudioDecodeError:
        return _decode_ffmpeg(data)


def pcm_from_bytes(chunk: bytes, encoding: str) -> np.ndarray:
    """Decode a raw streaming PCM chunk into an int16 array."""
    if encoding in ("pcm_s16le", "s16le", "pcm16", "linear16"):
        return np.frombuffer(chunk, dtype="<i2")
    if encoding in ("pcm_f32le", "f32le", "float32"):
        return np.frombuffer(chunk, dtype="<f4").astype(np.float32)
    raise AudioDecodeError(f"unsupported stream encoding: {encoding}")
