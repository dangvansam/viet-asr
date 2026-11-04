# @vietasr/web — WebAssembly / browser binding

Offline Vietnamese Speech AI SDK for the browser. Runs **fully on-device** — no server, no upload.

## Architecture

```
                  ┌──────────────────────────────────────────────┐
  audio (file /   │  JS  (src/index.js) — moves tensors only      │
  mic / stream)   └────────┬───────────────────────────┬─────────┘
        │                  │                           │
        ▼       ┌──────────▼───────────┐   ┌───────────▼─────────────┐
                │  WebAssembly          │   │  onnxruntime-web         │
                │  (vietasr-core.wasm)  │   │   encoder.int8.onnx      │
                │   - resampler         │   │   decoder.int8.onnx      │
                │   - Kaldi fbank       │   │   joiner.int8.onnx       │
                │   - beam search       │   │  (wasm EP; encoder may    │
                │   - BPE detokenize    │   │   use ["webgpu","wasm"])  │
                └───────────────────────┘   └──────────────────────────┘
```

Fbank, the modified beam search and detokenization are compiled from the **same C++**
as every native binding (`KaldiFbank`, `TransducerBeamSearch`, `PostProcessor`,
`Units`), so the browser produces the same transcript. The search is a step machine:
WASM asks for decoder contexts and joiner inputs, JS runs them through
onnxruntime-web, WASM consumes the outputs.

The model (~78 MB int8, three ONNX files) is fetched at runtime from the release
pinned on jsDelivr — override with `modelUrl`. Output already carries casing,
punctuation and written-form numbers.

## Install

```bash
npm install @viet-asr/web
```

## Usage

```js
import { Pipeline } from "@viet-asr/web";

const pipe = await Pipeline.create();

// decode an audio file to Float32 PCM with the Web Audio API
const ctx = new AudioContext();
const audioBuffer = await ctx.decodeAudioData(await file.arrayBuffer());
const pcm = audioBuffer.getChannelData(0);

const text = await pipe.transcribe(pcm, audioBuffer.sampleRate);
console.log(text);
```

See [examples/index.html](examples/index.html) for a complete file-picker demo.

## Build from source

Requires the [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html).

```bash
# activate emscripten
source /path/to/emsdk/emsdk_env.sh

# compile the DSP core to WASM + bundle the model chunks
cd bindings/webjs
./build-wasm.sh
# -> dist/vietasr-core.js + dist/vietasr-core.wasm + dist/model-config.js

# install the JS dependency
npm install
```

Serve `examples/index.html` over HTTP (WASM + `fetch` need a real origin):

```bash
npx serve .
# open http://localhost:3000/examples/
```

## API

| Method | Signature | Notes |
|---|---|---|
| `Pipeline.create` | `(options?) => Promise<Pipeline>` | fetches + verifies the chunked model, inits WASM + 3 ORT sessions |
| `pipeline.transcribe` | `(pcm: Float32Array, sampleRate?) => Promise<string>` | full clip; audio > 30 s is split at the quietest point |

`options`:

| Field | Default | |
|---|---|---|
| `modelUrl` | jsDelivr release | base URL with `chunks.json`, `tokens.txt`, `*.partNN` |
| `model` | — | preloaded `{encoder, decoder, joiner, tokens}` (skips fetching) |
| `maxActivePaths` | `4` | beam width; `1` = greedy |
| `executionProviders` | `["wasm"]` | encoder EPs, e.g. `["webgpu", "wasm"]` |

## Notes

- The WASM C ABI (`vietasr_wasm_*`) lives in [src/wasm_api.cc](src/wasm_api.cc).
- Serve the `.wasm` with `Content-Type: application/wasm` and enable cross-origin
  isolation (COOP/COEP) so onnxruntime-web can use threads + SIMD.
- The encoder dominates compute; the decoder/joiner calls are small (one joiner call
  per 40 ms frame, one decoder call per emitted token).
