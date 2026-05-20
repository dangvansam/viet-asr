# vad — Voice Activity Detection Module

Wraps Silero VAD (ONNX) for batch and streaming voice detection. Used by the `transcribe` preset to scope ASR work to voiced regions and by the `meeting` preset to feed diarization.

## Output field

```json
{
  "voiced_regions": [
    {"start": 0.4, "end": 2.1},
    {"start": 2.8, "end": 5.3}
  ]
}
```

## Hooks

| Hook | Action |
|---|---|
| `Init` | Loads Silero VAD ONNX model |
| `OnFrame` | Runs VAD on 32 ms frames, emits segment boundaries |
| `OnFinalize` | Flushes pending voiced region |

## Model

See [models.json](models.json).
