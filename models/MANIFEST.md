# Model Manifest

The `vietasr` model ships **inside** the SDK — its ONNX file is committed to this
repo as <50 MB chunks under `vietasr/` and reassembled at build time.
Other models (e.g. `vad`) are still hosted separately and downloaded on demand;
see [manifest.json](manifest.json).

## Modules

### vietasr (bundled)

- `model.onnx` — streaming Conformer, CTC head fused, int8-quantized (66.2 MB)
  - committed as chunks: `vietasr/model.onnx.part00`, `model.onnx.part01`
  - reassembly + integrity is described by `vietasr/chunks.json`
- `vocab.txt` — 4972 BPE tokens (64.6 KB)

Native builds embed the model directly into `libvietasr` (see
`core/cmake/EmbedModel.cmake`). The browser build (`@vietasr/web`) ships the
chunks inside the npm package. No network fetch, no cache directory.

To regenerate the chunks after producing a new `model.onnx`:

```
python scripts/split_model.py --model build/models/encoder.onnx \
    --out-dir models/vietasr --name model.onnx
```

### vad/1 (downloaded on demand)

- `silero_vad.onnx` — Silero VAD v5 (1.7 MB)

License: MIT (upstream). See <https://github.com/snakers4/silero-vad>.

## Cache directory

Downloaded models (currently only `vad`) land in:

- Linux: `~/.cache/vietasr/<module>/<version>/`
- macOS: `~/Library/Caches/vietasr/<module>/<version>/`
- iOS: `Library/Caches/vietasr/<module>/<version>/`
- Android: `context.cacheDir/vietasr/<module>/<version>/`
- Windows: `%LOCALAPPDATA%\vietasr\<module>\<version>\`

Override with the `VIETASR_MODEL_DIR` environment variable.
