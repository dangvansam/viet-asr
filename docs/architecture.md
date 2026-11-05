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
                                  │  Engine (ONNX Runtime) │
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

Bindings call `vietasr_*` functions from [core/include/vietasr.h](../core/include/vietasr.h). The ABI surface is small (about 30 functions), stable across module additions, and platform-agnostic. New modules surface as new JSON keys in the existing result string, never as new ABI symbols.

### Seam 2: C ABI ↔ Pipeline

The C ABI is a thin marshalling layer over `vietasr::Pipeline` ([core/include/vietasr/pipeline.h](../core/include/vietasr/pipeline.h)). Opaque pointers in, primitive types out. JSON strings carry rich results.

### Seam 3: Pipeline ↔ Module

The `Pipeline` walks audio and dispatches it to each registered `Module` ([core/include/vietasr/module.h](../core/include/vietasr/module.h)) through these hooks:

- `OnFrame` — raw audio (every chunk while streaming, the whole clip in batch)
- `OnSegment` — in batch mode, one segment spanning the whole clip
- `OnFinalize` — end-of-stream or end-of-file

`OnFeature`, `OnLogits` and `OnText` are declared for shared fbank features, shared encoder output and text post-processing, but the pipeline does not dispatch them yet: the `vietasr` module computes its own features and runs its own VAD-driven segmentation inside `OnFrame` / `OnFinalize`.

A module overrides only the hooks it cares about; the others are no-ops.

### Seam 4: Module ↔ Engine

`Engine` ([core/include/vietasr/engine.h](../core/include/vietasr/engine.h)) is an abstract interface; the shipped implementation wraps ONNX Runtime. Hardware acceleration (CUDA, DirectML, CoreML, NNAPI, XNNPACK …) is selected as an ONNX Runtime execution provider, so a module never knows where it runs.

### Seam 5: Module ↔ Preprocess / Utils

`preprocess/` owns audio-side shared work — feature extraction, ring buffering, resampling, WAV decoding. `utils/` owns generic plumbing — HTTP (libcurl / WinHTTP), MD5, JSON, filesystem, logger, model manager. Modules consume these but never duplicate them.

## Streaming vs non-streaming

Both modes invoke the same hooks. The only difference is who owns the loop:

| | Streaming | Non-streaming (batch) |
|---|---|---|
| Driver | `Session` (caller pushes chunks) | `Pipeline::TranscribeFile/Buffer` (engine pulls full audio) |
| Audio source | live PCM (mic, socket, file in chunks) | one WAV file or one buffer |
| Result polling | `Partial()` / `Result()` after each chunk | one JSON returned at the end |

Both paths drive `OnFrame → (OnSegment) → OnFinalize` on the same module code. No double-maintenance.

## Result schema

One JSON envelope. Each field is optional; a field is present when the module that writes it ran (and, for `segments` / `voiced_regions`, when the corresponding option is on). Bindings expose this either as raw JSON or as a typed struct depending on language conventions.

```json
{
  "text":           "Nửa vòng trái đất hơn bảy năm.",
  "partial":        "string (streaming only)",
  "is_final":       true,
  "segments":       [{"text": "...", "start": 0.0, "end": 2.4,
                      "words": [{"word": "Nửa", "start": 0.12, "end": 0.32}]}],
  "voiced_regions": [{"start": 0.0, "end": 2.4}],
  "speech":         {"detected": true, "voiced_ratio": 1.0,
                     "voiced_duration_s": 2.4, "total_duration_s": 2.4},
  "error":          "string (only when something failed)"
}
```

`segments` needs `emit_segments` (or `emit_words`, which also adds `words`); `voiced_regions` and `speech` need `emit_voiced_regions` or the `vad` module. The `diarization`, `gender`, `emotion`, `dialect`, `noise` and `punctuation` modules are placeholders today: they register and write fixed values (`speakers`, `gender`, `emotion`, `dialect`, `noise`, `punctuation`) so the schema and bindings are ready, but they run no model yet. Casing and punctuation already come from the ASR model itself.

## Model lifecycle

The **viet-asr** model — an int8 Zipformer2 transducer split into encoder, decoder and
joiner ONNX files plus `tokens.txt`, ~78 MB — is **not stored in the repository**. It
lives on Hugging Face ([dangvansam/viet-asr](https://huggingface.co/dangvansam/viet-asr)),
pinned by revision, size and checksum in
[core/src/modules/vietasr/models.json](../core/src/modules/vietasr/models.json).

When the `vietasr` module initialises it looks for the model in this order:

1. the module's `model_dir` config field
2. `$VIETASR_ASR_MODEL_DIR`
3. a copy embedded in `libvietasr` (builds with `-DVIETASR_EMBED_MODEL=ON`: Android, iOS / macOS
   xcframework)
4. the cache folder `<cache_dir>/vietasr/<revision>/`
5. otherwise it downloads the four files from Hugging Face into that cache folder
   (`ModelManager::EnsureBundle`, MD5-verified, written to a temp file then renamed)

If all of these fail — no network, `VIETASR_OFFLINE=1`, or `"download": false` — building
the pipeline returns an error naming the download URL; it never silently returns empty
transcripts. HTTP comes from libcurl on Linux / macOS and WinHTTP on Windows. Android and
iOS have no system HTTP client, so their build scripts embed the model instead: CMake
downloads it at configure time (SHA-256-checked) and links it into the library.

The encoder runs on the pipeline engine (and therefore on the selected execution provider);
the small decoder and joiner run on dedicated CPU sessions. The VAD models (Silero, TEN-VAD)
and the ITN grammar are small and always embedded.

```
cache_dir/
  vietasr/
    46adacf22d37/          # first 12 chars of the pinned Hugging Face revision
      encoder.int8.onnx
      decoder.int8.onnx
      joiner.int8.onnx
      tokens.txt
```

Cache root defaults:

- Linux: `$XDG_CACHE_HOME/vietasr/` or `~/.cache/vietasr/`
- macOS: `~/Library/Caches/vietasr/`
- Windows: `%LOCALAPPDATA%\vietasr\`

Override with `VIETASR_MODEL_DIR` (or `pipeline.set_model_dir(...)`). The Docker images
pre-fill this cache at image build, so containers run without network access.

## Threading

The pipeline is single-threaded per `Session`. Multiple sessions can run in parallel and share the loaded model weights. The model manager is thread-safe. Details: [thread-safety.md](thread-safety.md).

## Versioning

The C ABI follows semver. Module additions are minor versions. Module deletions or hook signature changes are major versions. The model is pinned to a Hugging Face revision in `models.json`; a new model ships by updating that file in a regular SDK release, and each revision gets its own cache folder.