# itn — Inverse Text Normalization

Rewrites the spoken-form ASR transcript into written form: numbers, dates,
currency, time, measurements, phone numbers, etc.

```
hai mươi nghìn đồng        ->  20000₫
lúc tám giờ ba mươi phút   ->  lúc 8:30
ngày hai mươi tháng năm    ->  ngày 20 tháng 5
năm phần trăm              ->  5 %
```

## How it works

A weighted finite-state transducer (WFST) pipeline ported from
[FunASR](https://github.com/modelscope/FunASR) (which adapted it from
[WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing), Apache-2.0):

1. **tag** — compose the input against the *tagger* grammar to label spans
   (`tokens { money { integer_part: "20000" currency: "₫" } }`).
2. **verbalize** — compose the tagged text against the *verbalizer* grammar to
   emit the written form.

The Vietnamese grammar is NeMo-derived: the tagger already emits token members
in verbalizer-ready order, so — unlike FunASR's WeText-style runtime — no
token-reorder pass is needed.

The grammars are compiled by OpenFst. Only the core `fst` library is built —
no command-line tools or extensions — so the dependency stays small and builds
to WebAssembly.

## Inputs / outputs

- **Hook**: `OnFinalize` — runs after the ASR module sets the transcript.
- **Reads**: `ResultBuilder::CurrentText()` (the canonical transcript).
- **Writes**:
  - overwrites the canonical `text` with the written form (when changed);
  - `itn` field: `{"applied":true,"lang":"vi","text":"..."}`.

```json
{ "text": "tôi có 20000₫", "itn": { "applied": true, "lang": "vi" } }
```

ITN expects lower-cased text without punctuation (raw ASR output). In the
`transcribe-rich` preset `itn` currently runs after `punctuation`; since the
punctuation module is still a stub this is harmless, but `itn` should move
before `punctuation` once that module does real work.

## Grammar files

`models/vi_itn_tagger.fst` (~540 KB) and `models/vi_itn_verbalizer.fst`
(~240 KB) are committed to the repo and **baked into libvietasr** at build time
(`VIETASR_EMBED_ITN=ON`) — nothing to download, works offline.

Regenerate them with [`scripts/gen_itn_fst.py`](../../../../scripts/gen_itn_fst.py)
when FunASR's `vi` grammar changes.

## Latency & size

- Grammar load: one-time at `Init`, a few ms.
- Per-utterance normalization: sub-millisecond for typical sentences.
- Added to libvietasr: ~780 KB of grammar + the static `fst` core library.

## Build options

| Option | Default | Effect |
|---|---|---|
| `VIETASR_MODULE_ITN` | `ON` | Build the module + fetch OpenFst |
| `VIETASR_EMBED_ITN`  | `ON` | Bake the grammar into libvietasr |

With `VIETASR_EMBED_ITN=OFF` the module compiles but passes text through
unchanged (no embedded grammar).
