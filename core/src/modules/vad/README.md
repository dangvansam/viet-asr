# vad — Voice Activity Detection Module

Detects voiced (speech) regions of an audio stream. Used by the `transcribe`
preset to scope ASR work and by the `meeting` preset to feed diarization.

The module is a thin dispatcher over three interchangeable **backends**, chosen
by the `backend` field of the module's JSON config.

## Backends

| `backend` | Detector | Model | Dependencies |
|---|---|---|---|
| `energy` | RMS-energy threshold | none | none — works on every build |
| `silero` | Silero VAD v5 | `silero_vad.onnx` (MIT) | ONNX Runtime |
| `tenvad` *(default)* | [TEN-VAD](https://github.com/TEN-framework/ten-vad) | `ten-vad.onnx` | ONNX Runtime |

Both neural models are **embedded into `libvietasr`** at build time — no
download, no network. When ONNX Runtime is not available, or a neural backend
fails to initialize, the module logs a warning and falls back to `energy`.

## Configuration

Every field is optional. Each backend ships its own recommended tuning, applied
first; any field you set in the config then overrides it — so SDK callers stay
free to change anything. An empty config (`{}`) yields the `tenvad` defaults.

`backend` selects the detector: `energy` \| `silero` \| `tenvad` (default
`tenvad`). The remaining fields default **per backend**:

| Field | Type | `energy` | `silero` | `tenvad` | Meaning |
|---|---|---|---|---|---|
| `threshold` | number | `0.5` | `0.43` | `0.41` | speech-probability cutoff, `[0, 1]` |
| `hangover_ms` | int | `200` | `250` | `250` | silence tolerated inside a region before it closes |
| `min_speech_ms` | int | `100` | `100` | `100` | regions shorter than this are dropped |
| `min_silence_ms` | int | `100` | `100` | `100` | regions closer than this gap are merged |
| `speech_pad_ms` | int | `30` | `500` | `500` | padding added to each side of every region |
| `hop_size` | int | — | — | `256` | TEN-VAD analysis hop in samples: `256` (16 ms) or `160` (10 ms) |

When the chosen backend is unavailable, the module falls back to `energy` and
re-applies the `energy` defaults above (keeping any fields you set explicitly).

```python
# Python — the config dict is passed through as JSON to every binding.
pipeline.add("vad", {"backend": "silero"})                    # silero defaults
pipeline.add("vad", {"backend": "tenvad", "threshold": 0.5})  # default + override
```

## Output fields

```json
{
  "speech": {
    "detected": true,
    "voiced_ratio": 0.87,
    "voiced_duration_s": 10.04,
    "total_duration_s": 11.52
  },
  "voiced_regions": [
    {"start": 0.48, "end": 2.53},
    {"start": 2.98, "end": 8.45}
  ]
}
```

- **`speech`** — the speech / no-speech classification. `detected` is the
  direct answer to "does this audio contain speech"; it is `true` when at least
  one voiced region was found. `voiced_ratio` is voiced time over total time.
- **`voiced_regions`** — the detected speech spans. `start` / `end` are seconds
  from the start of the stream; the array is empty when no speech is detected.

## Presets

The `vad` preset is a VAD-only pipeline — it classifies audio as speech or
no-speech with no ASR:

```python
result = vietasr.Pipeline.preset("vad").transcribe("clip.wav")
print(result["speech"]["detected"])   # True / False
```

It uses the `tenvad` backend (a real speech/non-speech classifier, unlike the
energy detector, which only distinguishes silence from sound).

## Streaming

The module works in both **batch** and **streaming** modes with the same
config. In streaming mode it updates the result live: after each
`accept_waveform(...)` call, `session.partial()` / `session.result()` already
carry the current `speech` classification and `voiced_regions` — you do not
have to wait for `session.final()`.

```python
with vietasr.Pipeline.preset("vad").stream(sample_rate=16000) as session:
    for chunk in microphone():
        session.accept(chunk)
        if session.partial()["speech"]["detected"]:
            print("speech…")
```

Regions evolve as audio arrives (an open region's `end` advances, and it may
later merge with the next) and are finalized by `session.final()`.

## Hooks

| Hook | Action |
|---|---|
| `Init` | Parse config, build the backend, load its embedded model |
| `OnFrame` | Buffer audio into fixed chunks, run the backend, refresh the live snapshot |
| `OnFinalize` | Flush the trailing chunk, emit the final `voiced_regions` + `speech` |

## Models

See [models.json](models.json). The Silero model is committed under
`core/models/vad/`; the TEN-VAD model under
`core/third_party/ten-vad/src/onnx_model/`. Both are baked in by
`core/cmake/EmbedVadModel.cmake` (toggle with `VIETASR_EMBED_SILERO` /
`VIETASR_EMBED_TENVAD`). The backends themselves can be excluded from the build
with `VIETASR_VAD_SILERO=OFF` / `VIETASR_VAD_TENVAD=OFF`.

## Attribution

- Silero VAD — MIT licensed, https://github.com/snakers4/silero-vad
- **Powered by ten-vad** — TEN-VAD is licensed under Apache-2.0 **with
  additional conditions** (a no-compete clause and an attribution request). See
  [`core/third_party/ten-vad/LICENSE`](../../third_party/ten-vad/LICENSE) and
  the Third-party notices section of the repository root `README.md`.
