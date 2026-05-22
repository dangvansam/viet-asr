# vietasr — Vietnamese ASR Module

Vietnamese ASR module: streaming Conformer encoder + CTC prefix beam search.

## Output fields

```json
{
  "text":     "tôi có 20000₫. lúc 8:30",
  "partial":  "lúc 8:30",
  "is_final": true,
  "segments": [
    {"text": "tôi có 20000₫", "start": 0.0,  "end": 1.84},
    {"text": "lúc 8:30",       "start": 2.40, "end": 3.52}
  ],
  "voiced_regions": [{"start": 0.0, "end": 1.84}, {"start": 2.40, "end": 3.52}],
  "speech": {"detected": true, "voiced_ratio": 0.86, ...}
}
```

- **`text`** — the running aggregate transcript of the whole stream, start to end.
- **`segments`** — one entry per sentence ("chunk"): `text` (ITN-normalized when
  `itn` is on) + `start` / `end` time. Emitted when `emit_segments` is on.
- **`voiced_regions` / `speech`** — emitted when `emit_voiced_regions` is on
  (see below).

## Configuration

Pass a JSON object to `pipeline.add("vietasr", {...})`:

| Field | Type | Default | Effect |
|---|---|---|---|
| `pause_punctuation` | bool | `true` | Insert sentence/clause punctuation at speech pauses |
| `sentence_pause_ms` | int | `1500` | Silence gap ≥ this between utterances → `.` |
| `clause_pause_ms` | int | `500` | Silence gap ≥ this (and `< sentence_pause_ms`) → `,` |
| `emit_segments` | bool | `false` | Emit the per-sentence `segments[]` array |
| `emit_voiced_regions` | bool | `false` | Emit `voiced_regions` + `speech` (from CTC-blank) |
| `itn` | bool | `false` | ITN-normalize each sentence as it ends |

The **`stream`** preset turns all three on — see below.

**Streaming sentences + per-sentence ITN.** The CTC endpoint detector commits a
sentence at every pause. With `emit_segments`, each committed sentence is added
to `segments[]` with its start/end time; with `itn`, that sentence is
inverse-text-normalized *the moment it ends* (`"hai mươi nghìn đồng"` →
`"20000₫"`). ITN touches only the committed text — the encoder/beam caches are
untouched, so the rest of the stream continues uninterrupted. `text` is the
running aggregate of all sentences. (`itn` is ignored if the `itn` module was
not built into the library.)

**Pause-driven punctuation.** The CTC endpoint detector already splits speech
into utterances at every pause. Each utterance records how long the silence
before it lasted; when the transcript is assembled, a long pause becomes `. `
and a shorter one `, ` (anything below `clause_pause_ms`, just a space):

```
"em đi học"  +  2.0 s pause  +  "anh ở nhà"   ->  "em đi học. anh ở nhà"
"hôm nay"    +  1.1 s pause  +  "trời đẹp"     ->  "hôm nay, trời đẹp"
```

Set `pause_punctuation: false` to disable it — utterances are then
newline-joined (legacy behavior). Note the endpoint detector only cuts on
silences ≳ 1 s, so this marks sentence and long-clause boundaries, not every
brief breath. It runs entirely inside the ASR step, so the punctuated text
flows straight into downstream modules such as `itn`.

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
