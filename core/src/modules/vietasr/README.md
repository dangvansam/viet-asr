# vietasr — Vietnamese ASR module

Offline **Zipformer2 transducer** (85.8M params, int8) with **built-in ITN, punctuation
and casing**: the model writes `"Ngân sách dưới 50 triệu."` directly — no separate
ITN or punctuation pass.

```
pcm 16 kHz ─► Silero VAD segmenter ─► Kaldi fbank (80) ─► encoder ─► modified beam search ─► text
              (adaptive pauses, ≤ 30 s)                     (ORT)     (decoder + joiner, 4 paths)
```

## Output fields

```json
{
  "text":     "Dạ, em chào anh. Hợp đồng của anh có giá trị 20 triệu.",
  "partial":  "Dạ, em chào anh. Hợp đồng của anh",
  "is_final": true,
  "segments": [
    {"text": "Dạ, em chào anh.", "start": 0.32, "end": 1.41},
    {"text": "Hợp đồng của anh có giá trị 20 triệu.", "start": 2.88, "end": 5.06}
  ],
  "voiced_regions": [{"start": 0.32, "end": 1.41}, {"start": 2.88, "end": 5.06}],
  "speech": {"detected": true, "voiced_ratio": 0.62}
}
```

## Configuration

Pass a JSON object to `pipeline.add("vietasr", {...})`:

| Field | Type | Default | Effect |
|---|---|---|---|
| `decoding_method` | string | `modified_beam_search` | `greedy_search` = 1 path |
| `max_active_paths` | int | `4` | Beam width |
| `num_threads` | int | `1` | Encoder intra-op threads |
| `partial_interval_ms` | int | `640` | Min audio between partial re-decodes (grows to 25 % of the utterance) |
| `emit_segments` | bool | `false` | Emit per-sentence `segments[]` |
| `emit_words` | bool | `false` | Add `words[]` (`word`, `start`, `end`) to every segment; implies `emit_segments` |
| `emit_voiced_regions` | bool | `false` | Emit `voiced_regions` + `speech` |
| `vad_backend` | string | `silero` | `silero` / `tenvad` / `energy` |
| `vad_threshold` | float | `0.5` | Speech probability cutoff |
| `min_silence_ms` | int | `1200` | Pause that ends a short utterance |
| `adaptive_silence` | bool | `true` | Shorten the required pause as the utterance grows |
| `adaptive_short_ms` / `adaptive_mid_silence_ms` | int | `4000` / `500` | After 4 s of speech, a 500 ms pause ends the utterance |
| `adaptive_long_ms` / `adaptive_long_silence_ms` | int | `10000` / `300` | After 10 s of speech, a 300 ms pause ends the utterance |
| `speech_pad_ms` | int | `1000` | Audio kept around each utterance |
| `max_utterance_ms` | int | `30000` | Longer speech is split at the quietest point of the last 3 s (capped at 38 s — the encoder accepts ≤ 40 s) |
| `model_dir` | string | — | Load `encoder/decoder/joiner(.int8).onnx` + `tokens.txt` from a folder |
| `download` | bool | `true` | Download the model from Hugging Face when no local copy is found |
| `itn` | bool | `false` | Extra rule-based ITN pass (rarely needed — the model already does ITN) |

Model lookup order: `model_dir` → `$VIETASR_ASR_MODEL_DIR` → embedded (Android / iOS builds) →
`<cache>/vietasr/<revision>/` → download from Hugging Face into that cache folder. When none
of them yields a model, building the pipeline fails with an error naming the download URL.

## Word timestamps

`{"emit_words": true}` adds per-word times taken from the transducer itself: a word
starts at the 40 ms encoder frame where its first BPE piece is emitted (identical to
sherpa-onnx token timestamps) and ends where the next word starts, capped at 1 s.

```json
{"text": "Đây là tháng", "start": 0.0, "end": 1.2,
 "words": [{"word": "Đây", "start": 0.0, "end": 0.64},
           {"word": "là", "start": 0.64, "end": 0.92},
           {"word": "tháng", "start": 0.92, "end": 1.16}]}
```

## Streaming

The model is non-causal, so streaming is **VAD-segmented**: each utterance is decoded
offline when its pause arrives, and while the speaker is still talking the current
utterance is re-decoded for the `partial`. The final transcript of a stream is
byte-identical to a batch run over the same audio.

## Files

- `vietasr.{h,cc}` — `Module` implementation, config, segment joining
- `transducer_model.{h,cc}` — encoder/decoder/joiner sessions + model asset resolution
- `model_fetcher.{h,cc}` — downloads the pinned Hugging Face model into the cache
- `models.json` — pinned model source (repo, revision, sizes, checksums)
- `transducer_search.{h,cc}` — modified beam search as a step machine (shared with the WASM binding)
- `transducer_decoder.{h,cc}` — drives the search with the native ONNX sessions
- `utterance_segmenter.{h,cc}` — VAD-driven utterance cutting, long-utterance splitting
- `post_processor.{h,cc}` — BPE detokenizer (`▁` → space, byte-fallback `<0xNN>`)
- `units.{h,cc}` — `tokens.txt` loader

## Model files

The model is not stored in the repository. [models.json](models.json) pins the Hugging Face
repo ([dangvansam/viet-asr](https://huggingface.co/dangvansam/viet-asr)), the revision, and every
file's size, SHA-256 and MD5. It is the single source for all three delivery paths:

- **Runtime download** (default): `model_fetcher.{h,cc}` downloads the files on first use into
  `<cache>/vietasr/<revision>/`, verifying MD5. HTTP comes from libcurl (Linux, macOS) or WinHTTP
  (Windows). The cache root is `~/.cache/vietasr` (Linux), `~/Library/Caches/vietasr` (macOS),
  `%LOCALAPPDATA%\vietasr` (Windows), or `$VIETASR_MODEL_DIR`.
- **Build-time embedding** (`-DVIETASR_EMBED_MODEL=ON`, used by the Android and Apple builds, which
  have no system HTTP client): `core/cmake/EmbedModel.cmake` downloads the files with SHA-256
  checks and bakes them into `libvietasr`. `-DVIETASR_MODEL_SOURCE_DIR=<folder>` embeds local files
  instead (offline builds).
- **Browser**: `bindings/webjs/build-wasm.sh` writes the URL and SHA-256 list into
  `dist/model-config.js`.

| File | Size |
|---|---|
| `encoder.int8.onnx` | 69.5 MB |
| `decoder.int8.onnx` | 4.3 MB |
| `joiner.int8.onnx` | 4.0 MB |
| `tokens.txt` | 8192 BPE tokens (cased, digits, punctuation, byte fallback) |

| Variable | Effect |
|---|---|
| `VIETASR_ASR_MODEL_DIR` | Folder with the four files; skips the download |
| `VIETASR_MODEL_DIR` | Cache root for downloaded models |
| `VIETASR_MODEL_BASE_URL` | Download from a mirror (same file names) |
| `VIETASR_OFFLINE=1` | Never download; fail if no local model is found |

To ship a new model: upload it to Hugging Face, then update `revision`, `base_url` and the
per-file `size` / `sha256` / `md5` in `models.json`.

## Measured speed

AMD EPYC 9334, 1 thread, int8, via the Python binding:

| Mode | RTF | Speed |
|---|---|---|
| Batch | 0.035 | 28× real-time |
| Streaming (with partials) | 0.15 | 6.7× real-time |
