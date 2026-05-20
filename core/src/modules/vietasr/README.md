# vietasr — Vietnamese ASR Module

Vietnamese ASR module: streaming Conformer encoder + CTC prefix beam search.

## Output fields

```json
{
  "text":     "xin chào bạn",
  "partial":  "xin chào",
  "is_final": true,
  "segments": [{"start": 0.0, "end": 1.2, "text": "xin chào bạn", "confidence": 0.94}]
}
```

## Pipeline contribution

| Hook | Action |
|---|---|
| `Init` | Loads the embedded `model.onnx` + `vocab.txt` from buffers baked into libvietasr. Builds CTC beam search + endpoint detector. |
| `OnFeature` | Buffers 43 input frames (≈430 ms), runs encoder, emits logits via `OnLogits` for downstream modules. |
| `OnLogits` | CTC prefix beam search (beam=10). Updates partial transcript. |
| `OnSegment` | Endpoint reached (rule1=5s / rule2=1s / rule3=20s). Emits final segment. |
| `OnText` | (consumes from downstream) |
| `OnFinalize` | Flushes remaining features, returns final transcript. |

## Files

- `vietasr.{h,cc}` — main `Module` impl
- `ctc_beam_search.{h,cc}` — port of WeNet's `ctc_prefix_beam_search.cc`
- `ctc_endpoint.{h,cc}` — port of WeNet's `ctc_endpoint.cc` (rule1/2/3)
- `post_processor.{h,cc}` — BPE detokenizer (`▁` → space, strip `<unk>`)
- `units.{h,cc}` — vocab loader for `vocab.txt`
- `vietasr_test.cc` — golden fixture tests against Python oracle

## Model

The model ships **inside the SDK** — no download, no cache. It is committed to
the repo as <50 MB chunks under `models/vietasr/` and baked into `libvietasr`
at build time (see `core/cmake/EmbedModel.cmake`).

| File | Size | Format |
|---|---|---|
| `model.onnx` | 66.2 MB | ONNX, streaming Conformer + CTC head fused, int8-quantized |
| `vocab.txt` | 64.6 KB | 4972 BPE tokens, one per line |

Regenerate the chunks after producing a new model:
`python scripts/split_model.py --model <model.onnx> --out-dir models/vietasr --name model.onnx`

## Latency

| Platform | Hardware | RTF |
|---|---|---|
| Linux x86_64 | Xeon 8 core | 0.08 |
| Android | Pixel 7 | 0.18 |
| iOS | iPhone 14 | 0.12 |
| Browser (WASM, ONNX) | Chrome M1 | 0.35 |

(RTF = real-time factor; lower is faster. RTF 0.1 = transcribes 10s of audio in 1s.)
