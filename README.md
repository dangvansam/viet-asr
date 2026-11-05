# VietASR

[![Release](https://img.shields.io/github/v/release/dangvansam/viet-asr)](https://github.com/dangvansam/viet-asr/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/viet-asr)](https://pypi.org/project/viet-asr/)
[![Docker](https://img.shields.io/docker/v/dangvansam/viet-asr?label=docker&sort=semver)](https://hub.docker.com/r/dangvansam/viet-asr)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-dangvansam%2Fviet--asr-yellow)](https://huggingface.co/dangvansam/viet-asr)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](#-license)

**Vietnamese speech in, publish-ready text out — in a single pass.**

```python
import vietasr
print(vietasr.Pipeline.preset("transcribe").transcribe("call.wav").text)
# → "Vâng, thì em gọi điện lại để tư vấn và hỗ trợ chị tốt hơn ạ."
```

## 🌟 Highlights

- **Already-written text** — casing, punctuation and numbers in written form come straight
  out of the model (`"Ngân sách dưới 50 triệu."`). No ITN step, no punctuation/casing (PnC)
  model, no post-processing chain.
- **One small model, one pass** — 86M parameters, 78 MB int8, baked into the SDK.
- **28× faster than real time** on a single CPU core, end to end.
- **Offline and private** — no API keys, no cloud; audio never leaves the device.
- **Batch and live streaming** — VAD-segmented captions with partials, word-level
  timestamps, any sample rate.
- **Runs on any platform** — Linux, Windows, macOS, Android, iOS and the browser (WASM),
  with Python, Node.js, Go, Rust, Java/Kotlin, C#, Swift, C/C++ bindings, a Docker server —
  or the raw ONNX files with sherpa-onnx.
- **Measured in the open** — WER on public Vietnamese test sets, every system scored on
  its raw formatted output ([benchmark](#-benchmark)).

---

## ⚡ Install

Every binding wraps the **same C++ core** and produces **byte-identical transcripts**, so
the API is the same everywhere. Current release: **v0.1.0**.

| Channel | Status |
|---|---|
| PyPI · Docker Hub · Go modules · Swift Package · GitHub Packages (Java, Android) | ✅ v0.1.0 |
| npm (`viet-asr`, `@viet-asr/web`) · crates.io · NuGet | ⏳ next release — install from git for now |

The platform native library (with the 78 MB int8 model baked in) is fetched once from the
matching [GitHub Release](https://github.com/dangvansam/viet-asr/releases) on install or
first use, then cached under `~/.cache/viet-asr/`. Set `VIETASR_NATIVE_DIR` to point at a
local build instead.

<details open>
<summary>🐍 <b>Python</b> — <code>pip install viet-asr</code></summary>

```bash
# from PyPI
pip install viet-asr

# from git
pip install "git+https://github.com/dangvansam/viet-asr.git#subdirectory=bindings/python"
```

Python 3.8+. The PyPI wheel bundles the native library and model — nothing else to fetch.
Installing from git compiles `core/` and needs CMake and a C++17 compiler.
</details>

<details>
<summary>🟢 <b>Node.js</b> — from git (npm coming)</summary>

```bash
git clone https://github.com/dangvansam/viet-asr
npm install ./viet-asr/bindings/nodejs
```

Node.js 16+. A `postinstall` step downloads the native library for your platform.
</details>

<details>
<summary>🐹 <b>Go</b> — <code>go get …/bindings/go</code></summary>

```bash
go get github.com/dangvansam/viet-asr/bindings/go
```

Go 1.18+. `go get` resolves straight from git — the same command serves a tagged release
(`@v0.1.0`) or the latest code (`@main`). Pure Go (purego, no cgo); the native library is
downloaded on first use.
</details>

<details>
<summary>🦀 <b>Rust</b> — from git (crates.io coming)</summary>

In `Cargo.toml`:

```toml
[dependencies]
viet-asr = { git = "https://github.com/dangvansam/viet-asr", package = "viet-asr" }
```

`build.rs` downloads the native library at build time.
</details>

<details>
<summary>☕ <b>Java / Kotlin</b> — <code>io.github.dangvansam:viet-asr</code></summary>

Gradle, from **GitHub Packages** (needs a GitHub token with `read:packages`):

```kotlin
repositories {
    maven {
        url = uri("https://maven.pkg.github.com/dangvansam/viet-asr")
        credentials {
            username = System.getenv("GITHUB_ACTOR")
            password = System.getenv("GITHUB_TOKEN")
        }
    }
}
dependencies {
    implementation("io.github.dangvansam:viet-asr:0.1.0")
}
```

Java 11+. The native library is downloaded on first use.
</details>

<details>
<summary>🔷 <b>C# / .NET</b> — from git (NuGet coming)</summary>

```bash
git clone https://github.com/dangvansam/viet-asr
dotnet pack viet-asr/bindings/csharp/Vietasr -o ./nupkg
dotnet add package viet-asr --source ./nupkg
```

.NET 8+. The native library is resolved and downloaded on first use.
</details>

<details>
<summary>🌐 <b>Browser</b> — from git (npm coming)</summary>

```bash
git clone https://github.com/dangvansam/viet-asr
cd viet-asr/bindings/webjs && npm run build
```

Needs Emscripten (`em++`). Ships the WASM core plus `onnxruntime-web`; the model is
fetched at runtime (override with the `modelUrl` option).
</details>

<details>
<summary>🤖 <b>Android</b> — <code>io.github.dangvansam:viet-asr-android</code></summary>

Gradle, from **GitHub Packages** (GitHub token with `read:packages`):

```kotlin
repositories {
    maven {
        url = uri("https://maven.pkg.github.com/dangvansam/viet-asr")
        credentials {
            username = System.getenv("GITHUB_ACTOR")
            password = System.getenv("GITHUB_TOKEN")
        }
    }
}
dependencies {
    implementation("io.github.dangvansam:viet-asr-android:0.1.0")
}
```

minSdk 24. The AAR bundles `arm64-v8a`, `armeabi-v7a` and `x86_64`. From git: clone, run
`bindings/android/build-jnilibs.sh`, then `./gradlew :lib:assembleRelease`.
</details>

<details>
<summary>🍎 <b>iOS / macOS</b> — Swift Package</summary>

Swift Package Manager — add the package and `import VietASR`:

```swift
.package(url: "https://github.com/dangvansam/viet-asr.git", from: "0.1.0")
```

iOS 13+ / macOS 11+ (runs on every release up to the latest). The package
ships `CVietASR.xcframework` (iOS device + simulator + macOS), with the ASR
model and ONNX Runtime baked in. From git:
clone, then `bash bindings/apple/build-xcframework.sh` and open
`bindings/apple/Package.swift`. See [bindings/apple/README.md](bindings/apple/README.md).
</details>

<details>
<summary>🐳 <b>Server</b> — <code>docker run dangvansam/viet-asr</code></summary>

```bash
docker run -p 8000:8000 dangvansam/viet-asr          # CPU, amd64 + arm64
docker run --gpus all -p 8000:8000 dangvansam/viet-asr:gpu   # NVIDIA CUDA
```

OpenAI-compatible `POST /v1/audio/transcriptions` plus `WebSocket /v1/stream`, with a
built-in web UI. See [server/README.md](server/README.md).
</details>

<details>
<summary>⚙️ <b>C / C++</b> — link <code>libvietasr</code></summary>

```bash
cmake -S core -B build && cmake --build build
```

Then link `libvietasr` and `#include <vietasr.h>`
([core/include/vietasr.h](core/include/vietasr.h)).
</details>

---

## ✨ Two modes, both first-class

### 📄 Batch — transcribe a whole file

```python
import vietasr

pipe = vietasr.Pipeline.preset("transcribe")
result = pipe.transcribe("meeting.wav")
print(result.text)
```

### 🔴 Streaming — live captions as you speak

```python
pipe = vietasr.Pipeline.preset("stream")

with pipe.stream(sample_rate=16000) as session:
    for chunk in microphone():
        session.accept(chunk)
        print(session.partial().text)   # grows while you speak
    print(session.final().text)
```

Speech is cut into sentences on pauses (embedded Silero VAD); each finished sentence is
decoded once and committed, while the sentence in progress is refreshed as a partial.
The final text of a stream is identical to a batch run over the same audio.

Any sample rate, mono or stereo — it resamples internally. Feed it 8 kHz phone audio or 48 kHz studio audio, it just works.

---

## 🧩 Modular pipeline — grow without rewriting

VietASR isn't just ASR. It's a **pipeline of plug-in modules**. Today it transcribes; tomorrow you add speaker diarization, emotion, dialect — without changing a line of your integration code.

```python
pipe = (vietasr.Pipeline.new()
    .add("vad")                                   # voice activity detection
    .add("vietasr", {"emit_segments": True})      # ASR — ITN + punctuation included
    .add("gender")                                # speaker gender (coming)
    .add("emotion")                               # tone (coming)
    .build())

r = pipe.transcribe("call.wav")
print(r.text, r["segments"])
```

One JSON result, optional fields — a field appears only if its module ran:

```json
{
  "text": "Ngân sách dưới 50 triệu. Em gọi lại để hỗ trợ chị tốt hơn ạ.",
  "segments": [
    {"start": 0.4, "end": 2.1, "text": "Ngân sách dưới 50 triệu."},
    {"start": 3.0, "end": 5.2, "text": "Em gọi lại để hỗ trợ chị tốt hơn ạ."}
  ]
}
```

### ⏱️ Word timestamps

Add `{"emit_words": true}` to get per-word start/end times straight from the transducer —
no forced aligner, no extra model:

```python
pipe = vietasr.Pipeline.new().add("vietasr", {"emit_words": True}).build()
for w in pipe.transcribe("call.wav")["segments"][0]["words"]:
    print(f'{w["start"]:.2f}–{w["end"]:.2f}  {w["word"]}')
```

Word starts land on the 40 ms encoder frame where each word is emitted, matching
sherpa-onnx token timestamps.

---

## 🎙️ Voice activity detection — is there speech?

Sometimes you don't want a transcript — you just need to know **whether audio
contains speech**: to gate a recorder, skip silent files, or scope ASR to the
voiced parts. The `vad` preset is a VAD-only pipeline, no ASR:

```python
r = vietasr.Pipeline.preset("vad").transcribe("clip.wav")

print(r["speech"]["detected"])      # True  — does the audio contain speech?
print(r["speech"]["voiced_ratio"])  # 0.84  — fraction of the clip that is speech
print(r["voiced_regions"])          # [{"start": 0.48, "end": 2.53}, ...]
```

Three interchangeable backends — pick one per call, no rebuild:

| backend | detector | |
|---|---|---|
| `tenvad` | TEN-VAD (neural) | accurate — the default; *Powered by ten-vad* |
| `silero` | Silero VAD v5 (neural) | accurate, MIT-licensed |
| `energy` | RMS energy | dependency-free, always available |

```python
pipe = (vietasr.Pipeline.new()
    .add("vad", {"backend": "tenvad", "threshold": 0.6, "min_speech_ms": 200})
    .build())
```

Every knob — `threshold`, `hangover_ms`, `min_speech_ms`, `min_silence_ms`,
`speech_pad_ms`, `hop_size` — is configurable; see the
[vad module docs](core/src/modules/vad/README.md). Both neural models are baked
into the SDK (no download), and it runs in **batch and streaming** alike —
`session.partial()` carries the live speech verdict as audio arrives.

---

## 📋 At a glance

| | VietASR |
|---|---|
| Output | Written-form text: casing, punctuation, numbers (ITN) — in-model |
| Model | Zipformer2 transducer, 86M params, 78 MB int8 |
| Speed | 28× real-time batch, 6.7× real-time streaming — one CPU core |
| Modes | Batch files and live streaming with partials |
| Extras | VAD (Silero / TEN-VAD / energy), word timestamps, any sample rate |
| Deployment | Offline, on-device; model embedded in the native library |
| Platforms | Linux · Windows · macOS · Android · iOS · Browser (WASM) · Docker |
| Bindings | C · C++ · Python · Node.js · Go · Rust · Java/Kotlin · C# · Swift · JS/WASM |
| License | Apache-2.0 (SDK and model) |

---

## 📊 Benchmark

Word error rate (WER, %) — lower is better. Every system is scored on its **raw output**
against cased, punctuated, written-form references, so missing punctuation, casing or ITN
counts as an error — this is the text a user actually reads.

### Public test sets

References were re-annotated in written form (casing, punctuation, numbers).

| System | Size | VIVOS | Bud500 | VLSP 2018 | VLSP 2021 | VLSP 2022 | VLSP 2023 |
|---|---|---|---|---|---|---|---|
| **viet-asr** | 86M | 20.92 | 18.81 | 15.97 | 18.20 | 23.19 | 27.90 |
| **viet-asr (Pro)** | – | 8.85 | 9.37 | 7.67 | 12.87 | 16.27 | 21.55 |
| Google Chirp 3 | API | 11.73 | 21.99 | 10.02 | 15.45 | 22.93 | 23.15 |
| Viettel ASR | API | 16.69 | 25.68 | 17.19 | 26.04 | 22.39 | 27.82 |
| Parakeet-CTC-0.6B-vi | 0.6B | 15.54 | 17.30 | 17.06 | 22.09 | 28.59 | 30.07 |
| Qwen3-ASR-1.7B | 1.7B | 24.04 | 22.94 | 17.92 | 20.70 | 28.97 | 33.11 |
| gipformer 1.5 RNNT | 65M | 29.89 | 26.86 | 26.90 | 22.85 | 30.44 | 34.23 |
| ZipFormer-30M (hynt) | 30M | 30.24 | 26.79 | 27.04 | 23.63 | 31.51 | 33.22 |
| Nemotron-3.5-ASR-Streaming | 0.6B | 17.10 | 18.14 | 19.89 | 23.21 | 29.59 | 32.74 |
| VietASR-68M (zzasdf) | 68M | 31.94 | 28.45 | 27.74 | 24.96 | 32.05 | 36.68 |
| ChunkFormer-CTC-Large | 110M | 29.99 | 27.58 | 27.97 | 26.32 | 34.20 | 37.03 |
| PhoWhisper-Large | 1.55B | 23.63 | 31.17 | 26.59 | 25.34 | 34.39 | 35.38 |
| Whisper-Large-V3 | 1.55B | 17.36 | 53.39 | 16.88 | 28.29 | 35.07 | 35.75 |

*viet-asr* is the public model in this repo and on Hugging Face; *viet-asr (Pro)* is a
higher-accuracy model that is not publicly released.

viet-asr scores are for the fp32 export decoded with modified beam search (4 paths). The int8
model shipped in the SDK matched fp32 within 0.15 WER on a 900-utterance cross-check,
and the SDK's C++ decoder matched sherpa-onnx's output on the same files.

<details>
<summary>Public test sets</summary>

| Set | Source | Utterances |
|---|---|---|
| VIVOS | [AILAB, VNU-HCM](https://ailab.hcmus.edu.vn/vivos) — read speech | 760 |
| Bud500 | [linhtran92/viet_bud500](https://huggingface.co/datasets/linhtran92/viet_bud500) — podcasts | 7,500 |
| VLSP 2018 | VLSP 2018 ASR test | 795 |
| VLSP 2021 | VLSP 2021 ASR tasks 1 + 2 | 1,893 |
| VLSP 2022 | VLSP 2022 ASR tasks 1 + 2 | 11,683 |
| VLSP 2023 | VLSP 2023 ASR public test | 2,111 |

</details>

### Speed

AMD EPYC 9334, **one thread**, int8, end-to-end (audio in → formatted text out):

| Mode | RTF | Speed |
|---|---|---|
| Batch (`transcribe`) | 0.035 | **28× real-time** |
| Streaming with live partials | 0.15 | 6.7× real-time |

**RTF** = real-time factor (lower = faster). All bindings call the same C core, so
throughput is the same in every language; scale a server by running sessions in
parallel (the int8 encoder gains little from extra intra-op threads).

---

## 🚀 Quickstart in every language

<details>
<summary><b>Node.js</b></summary>

```js
const { Pipeline } = require("viet-asr");
const pipe = await Pipeline.preset("transcribe");
console.log(pipe.transcribe("audio.wav").text);
```
</details>

<details>
<summary><b>Go</b></summary>

```go
pipe, _ := vietasr.PipelinePreset("transcribe")
defer pipe.Close()
r, _ := pipe.TranscribeFile("audio.wav")
fmt.Println(r.Text())
```
</details>

<details>
<summary><b>Rust</b></summary>

```rust
let pipe = vietasr::Pipeline::preset("transcribe")?;
println!("{}", pipe.transcribe_file("audio.wav")?.text());
```
</details>

<details>
<summary><b>Java / Kotlin</b></summary>

```java
try (Pipeline pipe = Pipeline.preset("transcribe")) {
    System.out.println(pipe.transcribe("audio.wav").text());
}
```
</details>

<details>
<summary><b>C#</b></summary>

```csharp
using var pipe = Pipeline.Preset("transcribe");
Console.WriteLine(pipe.Transcribe("audio.wav").Text);
```
</details>

<details>
<summary><b>Browser</b></summary>

```js
import { Pipeline } from "@viet-asr/web";
const pipe = await Pipeline.create();

// batch
console.log(await pipe.transcribe(pcmFloat32, 16000));

// streaming — partial captions as you speak ("bốn mươi" → "40" via on-device ITN)
pipe.startStream();
const partial = await pipe.pushStream(micChunk, sampleRate);
const final = await pipe.finishStream();
```
</details>

<details>
<summary><b>C</b></summary>

```c
#include <vietasr.h>
VietasrPipeline* pipe = vietasr_pipeline_preset("transcribe");
puts(vietasr_transcribe_file(pipe, "audio.wav"));
vietasr_pipeline_free(pipe);
```
</details>

---

## 🛠️ How it works

```
audio ─► resample 16k ─► Silero VAD ─► Kaldi fbank ─► Zipformer encoder ─► beam search ─► formatted text
                         (sentences)    (80 bins)      (ONNX Runtime)       (decoder + joiner)
```

- **One C++ core**, one stable C ABI ([core/include/vietasr.h](core/include/vietasr.h))
- **ONNX Runtime backend** — same engine on every platform; the encoder follows the selected execution provider (CPU / CUDA / CoreML / DirectML …)
- **Pluggable modules** — adding a capability is one folder, see [docs/adding-a-module.md](docs/adding-a-module.md)
- **Thread-safe** — one pipeline serves many concurrent streams, see [docs/thread-safety.md](docs/thread-safety.md)
- **Zero-config** — the 78 MB int8 model is baked into the SDK; or load another with `{"model_dir": ...}`

Full design: [docs/architecture.md](docs/architecture.md).

---

## 🗺️ Roadmap

**Shipping now**

- ✅ Streaming + batch Vietnamese ASR
- ✅ Voice activity detection — speech-or-not classification, 3 backends (Silero / TEN-VAD / energy)
- ✅ Built-in ITN, punctuation and casing — produced by the ASR model itself
- ✅ 10 bindings — C, C++, Python, Node.js, Go, Java/Kotlin, C#, Rust, WebAssembly, Swift
- ✅ Android native libraries (arm64 / armv7 / x86_64)
- ✅ iOS / macOS — Swift package + `.xcframework`
- ✅ Model bundled in the SDK, zero config
- ✅ Any sample rate, mono / stereo
- ✅ Adaptive VAD segmentation — waits longer on short phrases, cuts faster on long ones
- ✅ Word-level timestamps from the transducer
- ✅ Long audio of any length — automatic splitting at quiet points
- ✅ Thread-safe multi-stream
- ✅ GPU builds — CUDA (Linux / Windows), DirectML (Windows)
- ✅ OpenAI-compatible server + Docker images (CPU multi-arch, CUDA)
- ✅ Browser: live-microphone streaming

**Coming next**

- ⏳ npm, crates.io and NuGet packages
- ⏳ Android `.aar` on Maven Central
- ⏳ `diarization` module — who-spoke-when
- ⏳ `gender` / `emotion` / `dialect` / `noise` modules
- ⏳ `language-id` + `speaker-id` modules
- ⏳ CoreML / NNAPI execution providers on mobile
- ⏳ WebGPU backend for the browser

Want a module sooner? Open an issue — or build it yourself in <100 lines: [docs/adding-a-module.md](docs/adding-a-module.md).

---

## 📦 The model

**viet-asr** — Zipformer2 transducer, 85.8M parameters, int8, 8192-token
cased BPE vocabulary with digits and punctuation. Three ONNX files (encoder 69.5 MB,
decoder 4.3 MB, joiner 4.0 MB) plus `tokens.txt`.

It ships **inside the SDK**: committed to this repo as <50 MB chunks under
[models/vietasr/](models/vietasr/) and baked into the native library at build time.
Details: [models/MANIFEST.md](models/MANIFEST.md). Standalone int8 ONNX files (usable
with sherpa-onnx) are on Hugging Face: [dangvansam/viet-asr](https://huggingface.co/dangvansam/viet-asr)
(mirror: [coflar-ai/viet-asr](https://huggingface.co/coflar-ai/viet-asr)). License: Apache-2.0.

## 🙏 Credits

- Zipformer transducer architecture and training recipe from [k2-fsa/icefall](https://github.com/k2-fsa/icefall); decoding validated against [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- Multi-platform binding design inspired by [Vosk](https://github.com/alphacep/vosk-api)
- Inference via [ONNX Runtime](https://onnxruntime.ai)
- Voice activity detection via [Silero VAD](https://github.com/snakers4/silero-vad) (MIT)
  and **Powered by [ten-vad](https://github.com/TEN-framework/ten-vad)**

## 📄 License

Apache 2.0 for the SDK code and the bundled viet-asr model.

Bundled third-party components keep their own licenses — see
[Third-party notices](#third-party-notices) below. Note that the optional
`tenvad` VAD backend (TEN-VAD) carries **Apache-2.0 with additional
conditions**, including a no-compete clause; build with
`-DVIETASR_VAD_TENVAD=OFF` to exclude it.

### Third-party notices

VietASR incorporates the third-party components listed below. Each remains
under its own license.

#### TEN-VAD

- **Used by:** the `tenvad` backend of the `vad` module.
- **Source:** https://github.com/TEN-framework/ten-vad
- **Vendored at:** [`core/third_party/ten-vad/`](core/third_party/ten-vad/)
  (pinned commit in [`COMMIT`](core/third_party/ten-vad/COMMIT); local
  modifications documented in [`PATCHES.md`](core/third_party/ten-vad/PATCHES.md)).
- **License:** **Apache License 2.0 with additional conditions** — see
  [`core/third_party/ten-vad/LICENSE`](core/third_party/ten-vad/LICENSE) and
  [`core/third_party/ten-vad/NOTICES`](core/third_party/ten-vad/NOTICES).

  > ⚠️ TEN-VAD is **not** plain Apache-2.0. The additional conditions include a
  > clause prohibiting deployment "in a way that competes with Agora's
  > offerings", and a requested attribution. Review the LICENSE before
  > redistributing VietASR with the `tenvad` backend enabled. The backend can
  > be excluded from the build with `-DVIETASR_VAD_TENVAD=OFF`.

  **Powered by ten-vad.**

  TEN-VAD itself contains code derived from [LPCNet](https://github.com/xiph/LPCNet)
  (BSD-2-Clause / BSD-3-Clause) — see its `NOTICES` file.

#### Silero VAD

- **Used by:** the `silero` backend of the `vad` module.
- **Source:** https://github.com/snakers4/silero-vad
- **Bundled model:** [`core/models/vad/silero_vad.onnx`](core/models/vad/silero_vad.onnx)
  (Silero VAD v5).
- **License:** MIT.

#### ONNX Runtime

- **Used by:** the inference engine (all neural modules).
- **Source:** https://github.com/microsoft/onnxruntime
- **License:** MIT. Fetched at build time, not vendored.

#### OpenFst

- **Used by:** the `itn` module.
- **Source:** https://www.openfst.org (CMake-friendly fork:
  https://github.com/csukuangfj/openfst)
- **License:** Apache-2.0. Fetched at build time, not vendored.

---

<div align="center">

**Vietnamese speech in, publish-ready text out. Offline. Free. Fast.**

⭐ Star the repo if VietASR is useful to you.

</div>
