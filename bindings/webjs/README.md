# @viet-asr/web — WebAssembly / browser binding

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

The model (~78 MB int8, three ONNX files + `tokens.txt`) is fetched at runtime from
Hugging Face ([dangvansam/viet-asr](https://huggingface.co/dangvansam/viet-asr), pinned
revision, size + SHA-256 checked) — override with `modelUrl`. Output already carries casing,
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

### Streaming (live microphone)

For real-time transcription, feed audio chunks as they arrive — the transcript
grows with each call:

```js
const pipe = await Pipeline.create();
pipe.startStream();
// ... for each chunk of captured PCM:
const partial = await pipe.pushStream(pcm, sampleRate);  // transcript so far
// ... when the user stops:
const final = await pipe.finishStream();
```

[examples/index.html](examples/index.html) is a complete demo with a **live
microphone** mode (text appears as you speak) and a WAV file picker, both driven
by a Web Worker so the encoder never blocks the UI.

## Build from source

Requires the [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html).

```bash
# activate emscripten
source /path/to/emsdk/emsdk_env.sh

# compile the DSP core (fbank + beam search) to WASM
cd bindings/webjs
bash build-wasm.sh
# -> dist/vietasr-core.js + dist/vietasr-core.wasm + dist/model-config.js

# install the JS dependency (onnxruntime-web)
npm install
```

The demo fetches the model from Hugging Face (append `?model=<base URL>` to use a
mirror). Serve the **repository root** over HTTP (WASM, workers and `fetch` all need
a real origin):

```bash
# from the repo root
npx serve .
# open http://localhost:3000/bindings/webjs/examples/
```

The microphone needs a secure context — `localhost` qualifies.

## API

| Method | Signature | Notes |
|---|---|---|
| `Pipeline.create` | `(options?) => Promise<Pipeline>` | fetches + verifies the model, inits WASM + 3 ORT sessions |
| `pipeline.transcribe` | `(pcm: Float32Array, sampleRate?) => Promise<string>` | full clip; audio > 30 s is split at the quietest point |
| `pipeline.startStream` | `() => void` | begin a streaming session |
| `pipeline.pushStream` | `(pcm: Float32Array, sampleRate?) => Promise<string>` | feed a chunk; returns committed sentences + current partial |
| `pipeline.finishStream` | `() => Promise<string>` | decode the tail; returns the final transcript |

`options`:

| Field | Default | |
|---|---|---|
| `modelUrl` | Hugging Face, pinned revision | base URL holding `encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, `tokens.txt` |
| `model` | — | preloaded `{encoder, decoder, joiner, tokens}` (skips fetching) |
| `maxActivePaths` | `4` | beam width; `1` = greedy |
| `executionProviders` | `["wasm"]` | encoder EPs, e.g. `["webgpu", "wasm"]` |
| `ortModuleUrl` | — | explicit URL of the onnxruntime-web ESM module (needed inside a Web Worker) |
| `ortWasmPaths` | — | directory holding onnxruntime-web's own `.wasm` files |

Streaming commits a sentence after a 1.2 s pause (or at 30 s) and re-decodes the
sentence in progress for the partial, at most every 640 ms (growing with its length).
See [examples/asr-worker.js](examples/asr-worker.js) for a live-microphone demo in a
Web Worker.

## Notes

- The WASM C ABI (`vietasr_wasm_*`) lives in [src/wasm_api.cc](src/wasm_api.cc).
- Serve the `.wasm` with `Content-Type: application/wasm` and enable cross-origin
  isolation (COOP/COEP) so onnxruntime-web can use threads + SIMD.
- The encoder dominates compute; the decoder/joiner calls are small (one joiner call
  per 40 ms frame, one decoder call per emitted token).
