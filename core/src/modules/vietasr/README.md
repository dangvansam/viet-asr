# vietasr — Vietnamese ASR module

Offline **Zipformer2 transducer** (85.8M params, int8) with **built-in ITN, punctuation
and casing**: the model writes `"Ngân sách dưới 50 triệu."` directly — no separate
ITN or punctuation pass.

```
pcm 16 kHz ─► Silero VAD segmenter ─► Kaldi fbank (80) ─► encoder ─► modified beam search ─► text
              (pauses ≥ 1.2 s, ≤ 30 s)                     (ORT)     (decoder + joiner, 4 paths)
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
| `emit_voiced_regions` | bool | `false` | Emit `voiced_regions` + `speech` |
| `vad_backend` | string | `silero` | `silero` / `tenvad` / `energy` |
| `vad_threshold` | float | `0.5` | Speech probability cutoff |
| `min_silence_ms` | int | `1200` | Pause that ends an utterance |
| `speech_pad_ms` | int | `1000` | Audio kept around each utterance |
| `max_utterance_ms` | int | `30000` | Longer speech is split at the quietest point of the last 3 s |
| `model_dir` | string | — | Load `encoder/decoder/joiner(.int8).onnx` + `tokens.txt` from a folder instead of the embedded model |
| `itn` | bool | `false` | Extra rule-based ITN pass (rarely needed — the model already does ITN) |

Model lookup order: `model_dir` → `$VIETASR_ASR_MODEL_DIR` → embedded → `<cache>/vietasr/`.

## Streaming

The model is non-causal, so streaming is **VAD-segmented**: each utterance is decoded
offline when its pause arrives, and while the speaker is still talking the current
utterance is re-decoded for the `partial`. The final transcript of a stream is
byte-identical to a batch run over the same audio.

## Files

- `vietasr.{h,cc}` — `Module` implementation, config, segment joining
- `transducer_model.{h,cc}` — encoder/decoder/joiner sessions + model asset resolution
- `transducer_search.{h,cc}` — modified beam search as a step machine (shared with the WASM binding)
- `transducer_decoder.{h,cc}` — drives the search with the native ONNX sessions
- `utterance_segmenter.{h,cc}` — VAD-driven utterance cutting, long-utterance splitting
- `post_processor.{h,cc}` — BPE detokenizer (`▁` → space, byte-fallback `<0xNN>`)
- `units.{h,cc}` — `tokens.txt` loader

## Model files

Committed as <50 MB chunks under `models/vietasr/` and embedded into `libvietasr`:

| File | Size |
|---|---|
| `encoder.int8.onnx` | 69.5 MB |
| `decoder.int8.onnx` | 4.3 MB |
| `joiner.int8.onnx` | 4.0 MB |
| `tokens.txt` | 8192 BPE tokens (cased, digits, punctuation, byte fallback) |

Regenerate: `python scripts/split_model.py --model-dir <dir with *.onnx + tokens.txt> --out-dir models/vietasr`

## Measured speed

AMD EPYC 9334, 1 thread, int8, via the Python binding:

| Mode | RTF | Speed |
|---|---|---|
| Batch | 0.035 | 28× real-time |
| Streaming (with partials) | 0.15 | 6.7× real-time |
