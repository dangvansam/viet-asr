# vietasr — JVM (Java/Kotlin) desktop binding

Offline Vietnamese Speech AI SDK for desktop JVM applications. JNA wrapper over the same C ABI that powers every other vietasr binding — works on Linux, macOS, and Windows.

For **Android**, use [bindings/android](../android) instead (JNI + AAR).

## Install

### Gradle

```kotlin
dependencies {
    implementation("io.vietasr:vietasr:0.1.0")
}
```

### Maven

```xml
<dependency>
    <groupId>io.vietasr</groupId>
    <artifactId>vietasr</artifactId>
    <version>0.1.0</version>
</dependency>
```

The jar bundles a prebuilt `libvietasr.{so,dylib,dll}` + `libonnxruntime.{so,dylib,dll}` for your platform. JNA loads them at runtime.

## Quickstart

```java
import io.vietasr.Pipeline;
import io.vietasr.Result;

try (Pipeline pipe = Pipeline.preset("transcribe")) {
    Result result = pipe.transcribe("audio.wav");
    System.out.println(result.text());
}
```

Kotlin:

```kotlin
import io.vietasr.Pipeline

Pipeline.preset("transcribe").use { pipe ->
    println(pipe.transcribe("audio.wav").text())
}
```

## Streaming

```java
import io.vietasr.Pipeline;
import io.vietasr.Session;

try (Pipeline pipe = Pipeline.preset("transcribe");
     Session session = pipe.stream(16000f)) {
    for (short[] chunk : micChunks()) {
        session.accept(chunk);
        System.out.println(session.partial().text());
    }
    System.out.println("FINAL: " + session.finalResult().text());
}
```

## Custom pipeline

```java
try (Pipeline pipe = Pipeline.create()
        .add("vad")
        .add("vietasr")
        .add("punctuation")
        .add("gender")
        .add("emotion")
        .build()) {
    Result result = pipe.transcribe("call.wav");
    System.out.println(result.text());
    System.out.println(result.toJson());
}
```

## API

| Class | Method | Notes |
|---|---|---|
| `Pipeline` | `preset(String)` | static; downloads the model on first use |
| `Pipeline` | `create()` | static; empty pipeline for composition |
| `Pipeline` | `add(module)` / `add(module, jsonConfig)` | fluent |
| `Pipeline` | `setBackend(Backend)` | `AUTO`, `ONNX`, `COREML` |
| `Pipeline` | `setModelDir(path)` | advanced; load model from a directory instead of the embedded one |
| `Pipeline` | `build()` | finalise; auto-called by `preset()` |
| `Pipeline` | `transcribe(wavPath)` / `transcribe(short[], sampleRate)` | returns `Result` |
| `Pipeline` | `stream(sampleRate)` | returns `Session` |
| `Pipeline` | `listModules()` / `listPresets()` / `version()` | static |
| `Pipeline` | `close()` | implements `AutoCloseable` |
| `Session` | `accept(short[])` / `accept(float[])` | true if endpoint reached |
| `Session` | `partial()` / `result()` / `finalResult()` | returns `Result` |
| `Session` | `reset()` / `close()` | — |
| `Result` | `text()`, `partial()`, `isFinal()`, `toJson()` | typed accessors |

## Build from source

```bash
# point at a built core (see core/README)
ln -s ../../build-core/libvietasr.so          _native/
ln -s ../../build-core/.../libonnxruntime.so* _native/

# Gradle
gradle build

# or Maven
mvn package

# or plain javac
JDK=/path/to/jdk-17
$JDK/bin/javac -cp libs/jna-5.14.0.jar -d build/classes src/main/java/io/vietasr/*.java
```

Run an example:

```bash
$JDK/bin/java -cp "libs/jna-5.14.0.jar:build/classes:build/examples" \
    Quickstart audio.wav
```

## Thread safety

- One `Pipeline` can serve many concurrent `Session`s — each Session has independent state.
- See [docs/thread-safety.md](../../docs/thread-safety.md).

## Models

The vietasr model (int8 Zipformer transducer — encoder + decoder + joiner + `tokens.txt`,
~78 MB) is **not shipped in the package**. On first use the native `libvietasr` downloads it
from Hugging Face ([dangvansam/viet-asr](https://huggingface.co/dangvansam/viet-asr), pinned
revision, MD5-checked) into `~/.cache/vietasr/` (Linux), `~/Library/Caches/vietasr/`
(macOS) or `%LOCALAPPDATA%\vietasr\` (Windows); after that it runs fully offline.
Linux needs the system `libcurl` (`libcurl4`), which nearly every distribution has.

- `VIETASR_ASR_MODEL_DIR=/path/to/model` — use local files, no download
- `VIETASR_MODEL_BASE_URL=https://mirror/...` — download from a mirror
- `VIETASR_OFFLINE=1` — never download; building the pipeline fails if no model is found
