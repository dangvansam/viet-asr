# @vietasr/web — WebAssembly / browser binding

Offline Vietnamese Speech AI SDK for the browser. Runs **fully on-device** — no server, no upload.

## Architecture

The browser binding splits the pipeline across two runtimes:

```
                  ┌─────────────────────────────────────────┐
  audio (mic /    │  JS  (src/index.js)                     │
  file / stream)  │   - orchestration                       │
        │         │   - carries att_cache / cnn_cache       │
        ▼         └───────────┬─────────────────┬───────────┘
                              │                 │
              ┌───────────────▼──────┐   ┌──────▼────────────────┐
              │  WebAssembly         │   │  onnxruntime-web      │
              │  (vietasr-core.wasm) │   │  (model.onnx)       │
              │   - fbank front-end  │   │   - streaming         │
              │   - CTC beam search  │   │     Conformer encoder │
              │   - BPE detokenize   │   │                       │
              │   - resampler        │   │                       │
              └──────────────────────┘   └───────────────────────┘
              (compiled from the shared    (the same model.onnx
               C++ core — identical to      shipped to every other
               every other binding)        vietasr binding)
```

The model ships **inside this package** — `model.onnx` is committed to the repo
as <50 MB chunks and bundled under `dist/model/`. `Pipeline.create()`
reassembles them at load time. No network fetch, no HuggingFace dependency.

The WASM module is compiled from the **exact same C++** as the desktop/mobile
bindings (`FeaturePipeline`, `CtcBeamSearch`, `PostProcessor`, `Units`,
`AudioResampler`) — so decoding behaviour is bit-identical. Only the encoder
inference is delegated to `onnxruntime-web`, because ONNX Runtime already ships
a mature WASM build.

## Install

```bash
npm install @vietasr/web
```

## Usage

```js
import { Pipeline } from "@vietasr/web";

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
# -> dist/vietasr-core.js + dist/vietasr-core.wasm + dist/model/

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
| `Pipeline.create` | `(options?) => Promise<Pipeline>` | reassembles the bundled `model.onnx` chunks + `vocab.txt`, inits WASM + ORT |
| `pipeline.transcribe` | `(pcm: Float32Array, sampleRate?) => Promise<string>` | full-clip transcription |
| `pipeline.reset` | `() => void` | clears streaming state |

`Pipeline.create` loads the model bundled in the package. To supply your own,
pass `options.unitsText` (vocab string) and `options.encoderModel`
(`ArrayBuffer`/`Uint8Array` of an ONNX model) to skip the bundled load.

## What runs where

| Stage | Runtime |
|---|---|
| Resample → 16 kHz | WASM (`AudioResampler`) |
| 80-bin log-Mel fbank | WASM (`FeaturePipeline`) |
| Streaming Conformer encoder | onnxruntime-web (`model.onnx`) |
| CTC prefix beam search | WASM (`CtcBeamSearch`) |
| BPE detokenize | WASM (`PostProcessor` + `Units`) |

## Notes

- The WASM module exports a small C ABI (`vietasr_wasm_*`) defined in
  [src/wasm_api.cc](src/wasm_api.cc).
- `att_cache` / `cnn_cache` (the streaming Conformer state) are carried between
  chunks as `Float32Array`s on the JS side and fed back into each `session.run`.
- For lowest latency, serve the `.wasm` with `Content-Type: application/wasm`
  and enable cross-origin isolation so `onnxruntime-web` can use threads + SIMD.
