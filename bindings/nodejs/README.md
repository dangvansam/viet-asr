# vietasr — Node.js binding

Offline Vietnamese Speech AI SDK for Node.js. Pure FFI over the same C/C++ core that powers every other vietasr binding.

## Install

```bash
npm install vietasr
```

The package bundles a prebuilt `libvietasr.{so,dylib,dll}` + `libonnxruntime.{so,dylib,dll}` for your platform under `_native/`. No system dependencies.

## Quickstart

```js
const vietasr = require("vietasr");

const pipe = vietasr.Pipeline.preset("transcribe");
const result = pipe.transcribe("audio.wav");
console.log(result.text);
pipe.close();
```

## Streaming

```js
const vietasr = require("vietasr");

const pipe = vietasr.Pipeline.preset("transcribe");
const session = pipe.stream(16000);

for (const chunk of micChunks()) {
    session.accept(chunk);          // Int16Array, Float32Array, or Buffer
    process.stdout.write(`\r${session.partial().text}`);
}
console.log(`\n${session.final().text}`);
session.close();
pipe.close();
```

## Custom pipeline

```js
const vietasr = require("vietasr");

const pipe = vietasr.Pipeline.new()
    .add("vad")
    .add("vietasr")
    .add("punctuation")
    .add("gender")
    .add("emotion")
    .build();

const result = pipe.transcribe("call.wav");
console.log(result.text);
console.log(result.field("gender"));
console.log(result.field("emotion"));
pipe.close();
```

## CLI

The package installs a `vietasr` binary:

```bash
vietasr audio.wav
vietasr --preset analytics call.wav --pretty
vietasr --module vad --module vietasr --module gender audio.wav
```

## API

| Object | Method | Returns |
|---|---|---|
| `Pipeline` | `.preset(name)` | `Pipeline` |
| `Pipeline` | `.new()` | `Pipeline` |
| `Pipeline` | `.add(module, config?)` | `Pipeline` |
| `Pipeline` | `.setBackend("auto" \| "onnx" \| "coreml")` | `Pipeline` |
| `Pipeline` | `.setModelDir(path)` | `Pipeline` |
| `Pipeline` | `.build()` | `Pipeline` |
| `Pipeline` | `.transcribe(filePathOrInt16Array, sampleRate?)` | `Result` |
| `Pipeline` | `.stream(sampleRate)` | `Session` |
| `Pipeline` | `.close()` | void |
| `Session` | `.accept(Int16Array \| Float32Array \| Buffer)` | bool (true = endpoint reached) |
| `Session` | `.partial()` / `.result()` / `.final()` | `Result` |
| `Session` | `.reset()` / `.close()` | void |
| `Result` | `.text`, `.partial`, `.isFinal`, `.segments`, `.speakers` | typed |
| `Result` | `.field(key)` | any (extra JSON keys) |
| `Result` | `.toJson()` | string |
| module-level | `vietasr.listModules()`, `listPresets()`, `version()`, `setLogLevel(level)` | — |

## Thread safety

- One `Pipeline` can serve many concurrent `Session`s — each Session has its own state.
- For pure parallelism, prefer multiple Sessions on a shared Pipeline (engine loaded once).
- See [docs/thread-safety.md](../../docs/thread-safety.md) for the full guide.

## Audio format

The SDK accepts any sample rate and either mono or stereo (auto-converted internally). For best accuracy, feed 16 kHz mono 16-bit PCM directly.

## Models

The vietasr model is **bundled inside the SDK** — no download, no network. `model.onnx`
is committed to the repo as <50 MB chunks and baked into the native `libvietasr` that
ships with this package.
