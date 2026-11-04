# Model Manifest

The `vietasr` model ships **inside** the SDK — committed to this repo as <50 MB chunks
under `vietasr/` and embedded into `libvietasr` at build time. No download, no cache.

## vietasr — Zipformer-vi transducer (bundled)

| | |
|---|---|
| Architecture | Zipformer2 transducer (non-streaming encoder, stateless decoder, context 2) |
| Parameters | 85.8M |
| Precision | int8 (ONNX Runtime dynamic quantization) |
| Vocabulary | 8192 BPE tokens — cased, digits, punctuation, byte fallback |
| Output | Written-form Vietnamese: casing, punctuation and ITN built into the model |
| Features | 80-dim Kaldi fbank, 16 kHz, 25 ms / 10 ms, no dither, `snip_edges=false` |
| Decoding | modified beam search, 4 active paths |

| File | Size | Chunks |
|---|---|---|
| `encoder.int8.onnx` | 69.5 MB | `encoder.int8.onnx.part00`, `.part01` |
| `decoder.int8.onnx` | 4.3 MB | `decoder.int8.onnx.part00` |
| `joiner.int8.onnx` | 4.0 MB | `joiner.int8.onnx.part00` |
| `tokens.txt` | 104 KB | — |

License: Apache-2.0. Hugging Face: [dangvansam/viet-asr](https://huggingface.co/dangvansam/viet-asr)
(mirror [coflar-ai/viet-asr](https://huggingface.co/coflar-ai/viet-asr)) — int8 ONNX.

`vietasr/chunks.json` (manifest v2) records every chunk's size and SHA-256; the CMake
build verifies them before embedding (`core/cmake/EmbedModel.cmake`).

Regenerate after exporting a new model:

```
python scripts/split_model.py --model-dir <dir with encoder/decoder/joiner(.int8).onnx + tokens.txt> \
    --out-dir models/vietasr
python scripts/join_model.py --chunk-dir models/vietasr --out-dir /tmp/vietasr-model
```

To run a different model without rebuilding, point the module at a folder:
`pipeline.add("vietasr", {"model_dir": "/path"})` or set `VIETASR_ASR_MODEL_DIR`.

## vad (bundled)

- `silero_vad.onnx` — Silero VAD v5 (MIT), used by the `vad` module and by `vietasr`
  for utterance segmentation.
- `ten-vad.onnx` — TEN-VAD (Apache-2.0 with additional conditions), `vad` module only.

## Cache directory

Models fetched on demand land in:

- Linux: `~/.cache/vietasr/<module>/<version>/`
- macOS: `~/Library/Caches/vietasr/<module>/<version>/`
- iOS: `Library/Caches/vietasr/<module>/<version>/`
- Android: `context.cacheDir/vietasr/<module>/<version>/`
- Windows: `%LOCALAPPDATA%\vietasr\<module>\<version>\`

Override with the `VIETASR_MODEL_DIR` environment variable.
