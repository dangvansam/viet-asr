# vietasr — Rust binding

Offline Vietnamese Speech AI SDK for Rust. Safe wrapper over the same C ABI that powers every other vietasr binding.

## Install

```toml
[dependencies]
vietasr = "0.1.0"
```

The crate links `libvietasr` via `build.rs`. Point it at a built core with the `VIETASR_NATIVE_DIR` environment variable, or vendor the libraries under `_native/`.

## Quickstart

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = vietasr::Pipeline::preset("transcribe")?;
    let result = pipeline.transcribe_file("audio.wav")?;
    println!("{}", result.text());
    Ok(())
}
```

## Streaming

```rust
let pipeline = vietasr::Pipeline::preset("transcribe")?;
let mut session = pipeline.stream(16000.0)?;

for chunk in mic_chunks {
    session.accept_i16(&chunk);
    println!("{}", session.partial().text());
}
println!("FINAL: {}", session.finalize().text());
```

## Custom pipeline

```rust
let mut pipeline = vietasr::Pipeline::new();
pipeline
    .add("vad")?
    .add("vietasr")?
    .add("gender")?
    .add("emotion")?
    .build()?;

let result = pipeline.transcribe_file("call.wav")?;
println!("{}", result.text());
println!("{}", result.raw_json);
```

## API

| Item | Signature | Notes |
|---|---|---|
| `Pipeline::preset` | `(name: &str) -> Result<Pipeline>` | model bundled in the SDK |
| `Pipeline::new` | `() -> Pipeline` | empty pipeline |
| `Pipeline::add` | `(&mut self, module: &str) -> Result<&mut Self>` | fluent |
| `Pipeline::add_with_config` | `(&mut self, module: &str, json: &str) -> Result<&mut Self>` | |
| `Pipeline::set_backend` | `(&mut self, Backend) -> Result<&mut Self>` | `Auto/Onnx/CoreMl` |
| `Pipeline::set_model_dir` | `(&mut self, dir: &str) -> Result<&mut Self>` | |
| `Pipeline::build` | `(&mut self) -> Result<&mut Self>` | |
| `Pipeline::transcribe_file` | `(&self, path: &str) -> Result<TranscriptResult>` | |
| `Pipeline::transcribe_buffer` | `(&self, &[i16], f32) -> Result<TranscriptResult>` | |
| `Pipeline::stream` | `(&self, sample_rate: f32) -> Result<Session>` | |
| `Session::accept_i16` | `(&mut self, &[i16]) -> bool` | true if endpoint reached |
| `Session::accept_f32` | `(&mut self, &[f32]) -> bool` | |
| `Session::partial` / `result` | `(&self) -> TranscriptResult` | |
| `Session::finalize` | `(&mut self) -> TranscriptResult` | |
| `Session::reset` | `(&mut self)` | |
| `TranscriptResult::text` / `partial` | `(&self) -> String` | |
| `TranscriptResult::is_final` | `(&self) -> bool` | |
| `TranscriptResult::raw_json` | field | full JSON envelope |
| `list_modules` / `list_presets` | `() -> Vec<String>` | |
| `version` | `() -> String` | |

`Pipeline` and `Session` are `Send`; both free their native handle on `Drop`.

## Build from source

```bash
export VIETASR_NATIVE_DIR=/path/to/dir/with/libvietasr.so
cargo build --release --examples

./target/release/examples/quickstart audio.wav
./target/release/examples/streaming audio.wav
```

## Thread safety

- One `Pipeline` can serve many concurrent `Session`s — each Session has independent state.
- See [docs/thread-safety.md](../../docs/thread-safety.md).

## Models

The vietasr model is **bundled inside the SDK** — no download, no network. `model.onnx`
is committed to the repo as <50 MB chunks and baked into the native `libvietasr` this
crate links against.
