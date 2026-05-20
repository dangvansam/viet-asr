# vietasr — Python binding

Offline Vietnamese Speech AI SDK for Python. Built on top of the same C/C++ core that powers every other vietasr binding.

## Install

```bash
pip install vietasr
```

The wheel ships a prebuilt `libvietasr.{so,dylib,dll}` for your platform under `vietasr/_native/`. No system dependencies.

## Quickstart

```python
import vietasr

pipe = vietasr.Pipeline.preset("transcribe")
result = pipe.transcribe("audio.wav")
print(result.text)
```

## Streaming

```python
import vietasr
import numpy as np

pipe = vietasr.Pipeline.preset("transcribe")
with pipe.stream(sample_rate=16000) as session:
    for chunk in mic_chunks():
        session.accept(np.asarray(chunk, dtype=np.int16))
        print(session.partial().text)
    print("FINAL:", session.final().text)
```

## Custom pipeline

```python
pipe = (vietasr.Pipeline.new()
    .add("vad")
    .add("vietasr")
    .add("punctuation")
    .add("itn")
    .add("gender")
    .add("emotion")
    .build())

result = pipe.transcribe("call.wav")
print(result.text, result["gender"], result["emotion"])
```

## CLI

The wheel installs a `vietasr` command:

```bash
vietasr audio.wav
vietasr --preset analytics call.wav --pretty
vietasr --module vad --module vietasr --module gender audio.wav
```

## Build from source

```bash
git clone https://github.com/dangvansam/viet-asr
cd viet-asr/bindings/python
pip install -e .
```

`scikit-build-core` drives the CMake build under the hood. The compiled `libvietasr` is placed in `vietasr/_native/` and loaded by the CFFI wrapper.

## API reference

| Object | Method | Returns |
|---|---|---|
| `Pipeline` | `preset(name)` | `Pipeline` |
| `Pipeline` | `new()` | `Pipeline` |
| `Pipeline` | `add(module, config=None)` | `Pipeline` (self) |
| `Pipeline` | `set_backend(backend)` | `Pipeline` (self) |
| `Pipeline` | `set_model_dir(path)` | `Pipeline` (self) |
| `Pipeline` | `build()` | `Pipeline` (self) |
| `Pipeline` | `transcribe(source, sample_rate=16000)` | `Result` |
| `Pipeline` | `stream(sample_rate=16000)` | `Session` (context manager) |
| `Session` | `accept(pcm)` | bool |
| `Session` | `partial()` / `result()` / `final()` | `Result` |
| `Session` | `reset()` | None |
| `Result` | `.text`, `.partial`, `.is_final`, `.segments`, `.speakers` | typed |
| `Result` | `.field(key)` / `result[key]` | Any |
| `Result` | `.to_json()` | str |
