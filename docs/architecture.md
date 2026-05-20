# vietasr — Architecture

## Why a pipeline of modules?

Speech AI is not one thing. A real product wants ASR plus diarization plus emotion plus dialect plus punctuation plus inverse text normalization plus speaker identification plus noise classification. Each capability is a different model with different inputs and different outputs.

The naive design — one giant `Recognizer` class — collapses under that variety. Adding the seventh capability requires touching the previous six. Tests break. ABIs break. Bindings break.

The pipeline design keeps each capability in its own folder. Adding a module is one folder plus one line of registration. Existing modules are not edited, existing tests are not touched, and the public C ABI never changes.

## The five seams

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Bindings   │───►│   C ABI      │───►│   Pipeline   │
│  (Python,    │    │ (vietasr.h)  │    │ (orchestrator)│
│   Node, ...) │    │              │    │              │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                                │
                          ┌─────────────────────┼─────────────────────┐
                          ▼                     ▼                     ▼
                  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
                  │   Module A   │      │   Module B   │      │   Module N   │
                  └──────┬───────┘      └──────┬───────┘      └──────┬───────┘
                         │                     │                     │
                         └─────────────────────┼─────────────────────┘
                                               ▼
                                  ┌────────────────────────┐
                                  │  Engine (ONNX/CoreML)  │
                                  └────────────────────────┘
                                               │
                                               ▼
                                  ┌────────────────────────┐
                                  │   preprocess + utils   │
                                  │   (fbank, resampler,   │
                                  │   model_manager, ...)  │
                                  └────────────────────────┘
```

Each arrow is a seam. Each seam is a stable interface that lets one side change without the other noticing.

### Seam 1: Bindings ↔ C ABI

Bindings call `vietasr_*` functions from [core/include/vietasr.h](../core/include/vietasr.h). The ABI surface is small (about 25 functions), stable across module additions, and platform-agnostic. New modules surface as new JSON keys in the existing result string, never as new ABI symbols.

### Seam 2: C ABI ↔ Pipeline

The C ABI is a thin marshalling layer over `vietasr::Pipeline` ([core/include/vietasr/pipeline.h](../core/include/vietasr/pipeline.h)). Opaque pointers in, primitive types out. JSON strings carry rich results.

### Seam 3: Pipeline ↔ Module

The `Pipeline` walks audio frames and dispatches them to each registered `Module` ([core/include/vietasr/module.h](../core/include/vietasr/module.h)) through six hooks:

- `OnFrame` — raw audio
- `OnFeature` — shared post-fbank features (computed once, distributed)
- `OnLogits` — shared encoder output (computed once if an ASR-style module is in the pipeline)
- `OnSegment` — VAD or endpoint boundary
- `OnText` — text segment from any module that emits text
- `OnFinalize` — end-of-stream or end-of-file

A module overrides only the hooks it cares about. A gender classifier overrides `OnSegment` and `OnFinalize`. An ITN module overrides `OnText`. They are independent.

### Seam 4: Module ↔ Engine

`Engine` ([core/include/vietasr/engine.h](../core/include/vietasr/engine.h)) is an abstract interface with implementations for ONNX Runtime and (Apple-only) CoreML. A module never knows which backend it is running on. Swapping backends is one method on the pipeline.

### Seam 5: Module ↔ Preprocess / Utils

`preprocess/` owns audio-side shared work — feature extraction, ring buffering, resampling, WAV decoding. `utils/` owns generic plumbing — HTTP, MD5, JSON, filesystem, logger, model manager. Modules consume these but never duplicate them.

## Streaming vs non-streaming

Both modes invoke the same hooks. The only difference is who owns the loop:

| | Streaming | Non-streaming (batch) |
|---|---|---|
| Driver | `Session` (caller pushes chunks) | `Pipeline::TranscribeFile/Buffer` (engine pulls full audio) |
| Audio source | live PCM (mic, socket, file in chunks) | one WAV file or one buffer |
| Result polling | `Partial()` / `Result()` after each chunk | one JSON returned at the end |

Both paths drive `OnFrame → OnFeature → OnLogits → OnSegment → OnText → OnFinalize`. No double-maintenance.

## Result schema

One JSON envelope. Each field is optional. Presence of a field means the corresponding module ran. Bindings expose this either as raw JSON or as a typed struct depending on language conventions.

```json
{
  "text":      "string",
  "partial":   "string",
  "is_final":  true,
  "segments":  [{"start": 0.0, "end": 1.2, "text": "...",
                 "speaker": "S1", "confidence": 0.94}],
  "speakers":  [{"id": "S1", "total_time_s": 12.4}],
  "gender":    {"value": "M", "score": 0.91},
  "emotion":   {"value": "neutral", "score": 0.72},
  "dialect":   {"value": "north", "score": 0.88},
  "noise":     {"db": -22.1, "type": "clean"},
  "language":  "vi"
}
```

## Model lifecycle

The **vietasr** model is embedded directly into `libvietasr` at build time. It is
committed to the repo as <50 MB chunks under `models/vietasr/` and baked into the
binary by `core/cmake/EmbedModel.cmake` (see `core/src/embedded_model.cc`). No
download, no cache, no network.

Other models (e.g. `vad`) are still fetched on demand: on first use a pipeline asks
`ModelManager` to make sure each such module's files are present in the cache
directory, downloading what is missing and verifying MD5 against the manifest.

```
cache_dir/
  vietasr/
    vad/v1/silero.onnx
    diarization/v1/embedder.onnx
    ...
```

Cache root defaults:

- Linux: `$XDG_CACHE_HOME/vietasr/` or `~/.cache/vietasr/`
- macOS: `~/Library/Caches/vietasr/`
- iOS: `Library/Caches/vietasr/` inside the app container
- Android: `context.cacheDir/vietasr/`
- Windows: `%LOCALAPPDATA%\vietasr\`

Override with the env var `VIETASR_MODEL_DIR` for offline / airgapped deployments.

## Threading

The pipeline is single-threaded per `Session`. Multiple sessions can run in parallel and share the underlying model weights through the engine (engines are reference-counted under the pipeline). The model manager is thread-safe.

## Versioning

The C ABI follows semver. Module additions are minor versions. Module deletions or hook signature changes are major versions. Model bundles version independently (each `models.json` carries its own version), so a single binding release can pull in a newer model without changing code.