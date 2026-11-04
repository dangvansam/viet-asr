# VietASR

### Nói tiếng Việt — nhận văn bản chuẩn.

**Speak Vietnamese. Get publish-ready text.** Casing, punctuation and numbers come
straight out of the model — offline, on-device, one C++ core, nine language bindings.

```python
import vietasr
print(vietasr.Pipeline.preset("transcribe").transcribe("call.wav").text)
# → "Vâng, thì em gọi điện lại để tư vấn và hỗ trợ chị tốt hơn ạ."
```

No API keys, no cloud, no post-processing chain. The model ships inside the SDK.

## 🌟 Why this model

- **ITN built in** — numbers come out in written form (`"dưới 50 triệu"`,
  `"9 lần trong một thế kỷ"`, `"11, 12, 13."`). No rule-based ITN pass to tune.
- **Punctuation + casing built in** — `. , ? !` and proper capitalization are predicted
  by the same network that recognizes the words, from the audio itself.
- **Best formatted-text accuracy in our benchmark** — on 24 Vietnamese test sets it has
  the lowest WER on raw formatted output among all public models and commercial APIs we
  tested, **ahead of Google Chirp 3 and Viettel ASR**, and far ahead of open models 7–20×
  its size
  ([benchmark](#-benchmark)).
- **Strong on telephony** — 20.6 % formatted WER on call-center audio vs 24.0 % (Chirp 3)
  and 26.7 % (Viettel).
- **Small and fast** — 86M params, 78 MB int8, **28× real-time on one CPU core**.

---

## ⚡ Install

Every binding installs from its **package registry** or **directly from git** — both are
shown per language below. Each wraps the **same C++ core** and produces **byte-identical
transcripts**, so the API is the same everywhere.

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
<summary>🤖 <b>Android</b> — <code>io.github.dangvansam:viet-asr-android</code></summary>

Gradle, from **GitHub Packages**:

```kotlin
repositories {
    // add your GitHub username + token under credentials { }
    maven { url = uri("https://maven.pkg.github.com/dangvansam/viet-asr") }
}
dependencies {
    implementation("io.github.dangvansam:viet-asr-android:0.1.0")
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

## 🆚 Why VietASR?

| | ☁️ Cloud APIs | 🐋 Whisper-class | VietASR |
|---|---|---|---|
| Works offline | ❌ | ✅ | ✅ |
| Formatted output (ITN + punctuation + casing) | ✅ | partial | ✅ **in-model** |
| Live captions | ✅ | ❌ batch only | ✅ VAD-segmented |
| Cost per hour | 💰💰💰 | free | **free** |
| Privacy (audio leaves device) | ❌ | ✅ | ✅ |
| Model size | — | 0.8–3 GB | **78 MB** |
| Vietnamese tuned | generic | generic | **purpose-built** |
| Runs in a browser | ❌ | hard | ✅ WASM |
| Languages / bindings | SDK-limited | Python-first | **9 bindings** |
| Setup | API keys, billing | pip + CUDA | **one line** |

---

## 📊 Benchmark

Pooled WER (%) over **24 Vietnamese test sets** — public (VIVOS, VLSP 2018–2023,
Bud500, 7 regional-accent sets) plus private call-center and meeting sets. Lower is
better. Every system is scored on its **raw output** against cased, punctuated,
written-form references (`pnc`), so missing punctuation, casing or ITN counts as error —
which is what a user actually reads.

| System | Size | Benchmark | Regional | VLSP | Call center | Meeting | **All 24** |
|---|---|---|---|---|---|---|---|
| **VietASR Zipformer-vi** | 86M | 18.13 | 16.82 | **25.90** | **20.58** | **32.64** | **24.14** |
| Google Chirp 3 | API | 17.78 | 16.70 | 26.26 | 24.04 | 34.84 | 25.16 |
| Viettel ASR | API | 22.57 | 15.15 | 26.80 | 26.71 | 36.31 | 26.38 |
| Fun-ASR-MLT-Nano-2512 | ~0.8B | 14.56 | 14.09 | 30.09 | 34.39 | 44.23 | 29.57 |
| Parakeet-CTC-0.6B-vi | 0.6B | 17.10 | 16.41 | 30.54 | 37.59 | 43.97 | 30.80 |
| Qwen3-ASR-1.7B | 1.7B | 21.56 | 24.90 | 31.99 | 32.68 | 40.54 | 31.49 |
| gipformer 1.5 RNNT | 65M | 27.09 | 31.18 | 32.13 | 31.16 | 38.82 | 32.19 |
| ZipFormer-30M (hynt) | 30M | 27.11 | 31.20 | 32.90 | 33.38 | 40.95 | 33.25 |
| Nemotron-3.5-ASR-Streaming | 0.6B | 18.57 | 18.06 | 35.24 | 40.77 | 44.08 | 34.11 |
| VietASR-68M (zzasdf) | 68M | 28.49 | 32.09 | 34.19 | 34.38 | 42.10 | 34.43 |
| ChunkFormer-CTC-Large | 110M | 27.87 | 31.94 | 35.66 | 40.47 | 44.62 | 36.52 |
| PhoWhisper-Large | 1.55B | 29.31 | 29.33 | 38.94 | 38.33 | 47.06 | 37.98 |
| Whisper-Large-V3 | 1.55B | 40.22 | 20.02 | 49.15 | 36.53 | 49.73 | 43.28 |

**Word recognition only.** With everything lower-cased, punctuation stripped and digits
spelled out (`spoken_canonical`), the comparison measures bare word accuracy. Here the
models that output lower-case, unpunctuated text lead — they leave ITN and punctuation to
you:

| System | All 24 (`spoken_canonical`) |
|---|---|
| gipformer 1.5 RNNT | 12.09 |
| gipformer 1.0 RNNT | 13.61 |
| ZipFormer-30M (hynt) | 13.68 |
| Viettel ASR | 14.21 |
| Google Chirp 3 | 15.28 |
| VietASR-68M (zzasdf) | 15.34 |
| **VietASR Zipformer-vi** | **17.04** |
| ChunkFormer-CTC-Large | 17.34 |
| Qwen3-ASR-1.7B | 17.37 |
| PhoWhisper-Medium | 20.92 |
| Fun-ASR-MLT-Nano-2512 | 21.22 |
| Parakeet-CTC-0.6B-vi | 22.12 |
| Whisper-Large-V3 | 34.01 |

Scores are for the fp32 export decoded with modified beam search (4 paths). The int8
model shipped in the SDK matched fp32 within 0.15 WER on a 900-utterance cross-check,
and the SDK's C++ decoder matched sherpa-onnx's output on the same files.

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
- ✅ 9 bindings — C, C++, Python, Node.js, Go, Java/Kotlin, C#, Rust, WebAssembly
- ✅ Android native libraries (arm64 / armv7 / x86_64)
- ✅ Model bundled in the SDK, zero config
- ✅ Any sample rate, mono / stereo
- ✅ Endpoint-aware segmentation
- ✅ Thread-safe multi-stream

**Coming next**

- ⏳ iOS / macOS — Swift package + `.xcframework`
- ⏳ Android `.aar` on Maven Central
- ⏳ `diarization` module — who-spoke-when
- ⏳ `gender` / `emotion` / `dialect` / `noise` modules
- ⏳ `language-id` + `speaker-id` modules
- ⏳ GPU execution providers (CUDA / CoreML / NNAPI)
- ⏳ WebGPU backend for the browser

Want a module sooner? Open an issue — or build it yourself in <100 lines: [docs/adding-a-module.md](docs/adding-a-module.md).

---

## 📦 The model

**VietASR Zipformer-vi** — Zipformer2 transducer, 85.8M parameters, int8, 8192-token
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

Apache 2.0 for the SDK code and the bundled VietASR Zipformer-vi model.

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

**Nói tiếng Việt — nhận văn bản chuẩn. Offline. Free. Fast.**

⭐ Star it if VietASR saved you a cloud bill.

</div>
