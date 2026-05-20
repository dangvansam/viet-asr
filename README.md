# VietASR

### Offline Vietnamese speech-to-text. One model. Nine language bindings. Zero config.

Drop it in, write **three lines**, get a transcript. No API keys, no cloud, no network — the model runs **on-device**, everywhere from a Raspberry Pi to a browser tab.

```python
import vietasr
print(vietasr.Pipeline.preset("transcribe").transcribe("audio.wav").text)
# → "xin chào, bạn nghe rõ không"
```

That's the whole thing. The model ships inside the SDK — nothing to download. You're done.

---

## ⚡ Install

Every binding installs from its **package registry** or **directly from git** — both are
shown per language below. Each wraps the **same C++ core** and produces **byte-identical
transcripts**, so the API is the same everywhere.

The platform native library (with the 66 MB model baked in) is fetched once from the
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
<summary>🟢 <b>Node.js</b> — <code>npm install viet-asr</code></summary>

```bash
# from npm
npm install viet-asr

# from git
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
or the latest branch (append `@sdk`). Pure Go (purego, no cgo); the native library is
downloaded on first use.
</details>

<details>
<summary>🦀 <b>Rust</b> — <code>cargo add viet-asr</code></summary>

```bash
# from crates.io
cargo add viet-asr
```

From git, in `Cargo.toml`:

```toml
[dependencies]
viet-asr = { git = "https://github.com/dangvansam/viet-asr", package = "viet-asr" }
```

`build.rs` downloads the native library at build time.
</details>

<details>
<summary>☕ <b>Java / Kotlin</b> — <code>io.github.dangvansam:viet-asr</code></summary>

Gradle — from **GitHub Packages** (needs a GitHub token) or **JitPack** (no auth):

```kotlin
repositories {
    // GitHub Packages — add your GitHub username + token under credentials { }
    maven { url = uri("https://maven.pkg.github.com/dangvansam/viet-asr") }
    // JitPack — no authentication required
    maven { url = uri("https://jitpack.io") }
}
dependencies {
    implementation("io.github.dangvansam:viet-asr:0.1.0")    // GitHub Packages
    // implementation("com.github.dangvansam:viet-asr:<tag>") // JitPack
}
```

Java 11+. The native library is downloaded on first use.
</details>

<details>
<summary>🔷 <b>C# / .NET</b> — <code>dotnet add package viet-asr</code></summary>

```bash
# from NuGet
dotnet add package viet-asr
```

From git: clone the repo and `dotnet pack bindings/csharp/Vietasr`. .NET 8+. The native
library is resolved and downloaded on first use.
</details>

<details>
<summary>🌐 <b>Browser</b> — <code>npm install @viet-asr/web</code></summary>

```bash
# from npm
npm install @viet-asr/web
```

From git: clone the repo, then `cd bindings/webjs && npm run build`. Ships the WASM core
plus `onnxruntime-web`; the model is fetched at runtime (override with the `modelUrl`
option).
</details>

<details>
<summary>🤖 <b>Android</b> — <code>io.github.dangvansam:viet-asr</code></summary>

Gradle, from **GitHub Packages**:

```kotlin
repositories {
    // add your GitHub username + token under credentials { }
    maven { url = uri("https://maven.pkg.github.com/dangvansam/viet-asr") }
}
dependencies {
    implementation("io.github.dangvansam:viet-asr:0.1.0")
}
```

minSdk 24. The AAR bundles `arm64-v8a`, `armeabi-v7a` and `x86_64`. From git: clone, run
`bindings/android/build-jnilibs.sh`, then `./gradlew :lib:assembleRelease`.
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
pipe = vietasr.Pipeline.preset("transcribe")

with pipe.stream(sample_rate=16000) as session:
    for chunk in microphone():
        session.accept(chunk)
        print(session.partial().text)   # grows in real time
    print(session.final().text)
```

Any sample rate, mono or stereo — it resamples internally. Feed it 8 kHz phone audio or 48 kHz studio audio, it just works.

---

## 🧩 Modular pipeline — grow without rewriting

VietASR isn't just ASR. It's a **pipeline of plug-in modules**. Today it transcribes; tomorrow you add speaker diarization, emotion, dialect — without changing a line of your integration code.

```python
pipe = (vietasr.Pipeline()
    .add("vad")          # voice activity detection
    .add("vietasr")      # Vietnamese transcription
    .add("punctuation")  # . , ? !
    .add("itn")          # "hai mươi" → "20"
    .add("gender")       # speaker gender
    .add("emotion")      # tone
    .build())

r = pipe.transcribe("call.wav")
print(r.text, r["gender"], r["emotion"])
```

One JSON result, optional fields — a field appears only if its module ran:

```json
{
  "text": "tôi muốn đặt vé ngày 20 tháng 5",
  "segments": [{"start": 0.0, "end": 2.3, "text": "...", "speaker": "S1"}],
  "gender":  {"value": "M", "score": 0.91},
  "emotion": {"value": "neutral", "score": 0.72}
}
```

---

## 🆚 Why VietASR?

| | ☁️ Cloud APIs | 🐋 Whisper | VietASR |
|---|---|---|---|
| Works offline | ❌ | ✅ | ✅ |
| Streaming / live | partial | ❌ batch only | ✅ true streaming |
| Cost per hour | 💰💰💰 | free | **free** |
| Privacy (audio leaves device) | ❌ | ✅ | ✅ |
| Model size | — | 0.5–3 GB | **66 MB** |
| Vietnamese tuned | generic | generic | **purpose-built** |
| Runs in a browser | ❌ | hard | ✅ WASM |
| Languages / bindings | SDK-limited | Python-first | **9 bindings** |
| Setup | API keys, billing | pip + CUDA | **one line** |

Built on a **streaming Conformer + CTC** architecture — the kind that powers real-time captioning in production messaging apps.

---

## 📊 Benchmark

7.5 s Vietnamese clip, single CPU core, end-to-end (audio in → text out):

| Binding | Wall time | RTF | Speed |
|---|---|---|---|
| C / C++ | 0.90 s | 0.12 | **8× real-time** |
| Go | 0.91 s | 0.12 | 8× real-time |
| Java | 0.93 s | 0.12 | 8× real-time |
| Python | 1.00 s | 0.13 | 8× real-time |
| C# / .NET | 1.00 s | 0.13 | 8× real-time |
| Node.js | 1.03 s | 0.14 | 7× real-time |
| Browser (WASM) | 1.51 s | 0.20 | 5× real-time |

**RTF** = real-time factor (lower = faster). RTF 0.12 means 10 s of audio transcribes in 1.2 s. Streaming latency: a partial caption every ~320 ms.

Model: **66 MB**, int8-quantized, 4972-token Vietnamese BPE vocab. Fits in memory on a phone.

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
console.log(await pipe.transcribe(pcmFloat32, 16000));
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
audio ─► resample 16k ─► fbank ─► Conformer encoder ─► CTC beam search ─► text
                                  (ONNX Runtime)       (endpoint-aware)
```

- **One C++ core**, one stable C ABI ([core/include/vietasr.h](core/include/vietasr.h))
- **ONNX Runtime backend** — same engine on every platform
- **Pluggable modules** — adding a capability is one folder, see [docs/adding-a-module.md](docs/adding-a-module.md)
- **Thread-safe** — one pipeline serves many concurrent streams, see [docs/thread-safety.md](docs/thread-safety.md)
- **Zero-config** — the 66 MB model is baked into the SDK; no download, no network

Full design: [docs/architecture.md](docs/architecture.md).

---

## 🗺️ Roadmap

**Shipping now**

- ✅ Streaming + batch Vietnamese ASR
- ✅ 9 bindings — C, C++, Python, Node.js, Go, Java/Kotlin, C#, Rust, WebAssembly
- ✅ Android native libraries (arm64 / armv7 / x86_64)
- ✅ Model bundled in the SDK, zero config
- ✅ Any sample rate, mono / stereo
- ✅ Endpoint-aware segmentation
- ✅ Thread-safe multi-stream

**Coming next**

- ⏳ iOS / macOS — Swift package + `.xcframework`
- ⏳ Android `.aar` on Maven Central
- ⏳ `punctuation` module — restore `. , ? !`
- ⏳ `itn` module — inverse text normalization (`"hai mươi"` → `20`)
- ⏳ `diarization` module — who-spoke-when
- ⏳ `gender` / `emotion` / `dialect` / `noise` modules
- ⏳ `language-id` + `speaker-id` modules
- ⏳ GPU execution providers (CUDA / CoreML / NNAPI)
- ⏳ WebGPU backend for the browser

Want a module sooner? Open an issue — or build it yourself in <100 lines: [docs/adding-a-module.md](docs/adding-a-module.md).

---

## 📦 The model

Vietnamese streaming Conformer encoder — `model.onnx`, 66 MB, int8-quantized.

It ships **inside the SDK**: committed to this repo as <50 MB chunks under
[models/vietasr/](models/vietasr/) and baked into the native library at build time.
No download, no network, no cloud.

## 🙏 Credits

- Streaming Conformer + CTC architecture inspired by [WeNet](https://github.com/wenet-e2e/wenet)
- Multi-platform binding design inspired by [Vosk](https://github.com/alphacep/vosk-api)
- Inference via [ONNX Runtime](https://onnxruntime.ai)

## 📄 License

Apache 2.0 for the SDK code. The model has separate terms — see [models/MANIFEST.md](models/MANIFEST.md).

---

<div align="center">

**Made for Vietnamese developers. Offline. Free. Fast.**

⭐ Star it if VietASR saved you a cloud bill.

</div>
