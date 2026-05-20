# vietasr — Android binding

Offline Vietnamese Speech AI SDK for Android. JNI wrapper over the same C/C++ core that powers every other vietasr binding.

## Status

**Native libraries cross-compiled and staged.** The Java + JNI surface is complete; the Gradle module needs to be built once on a host with the Android SDK installed to produce the final `.aar`.

Pre-built native libs (this repo, ready to ship):

```
lib/src/main/jniLibs/
├── arm64-v8a/
│   ├── libvietasr.so       5.8 MB   (built with NDK r26d, API 24+)
│   └── libonnxruntime.so   27 MB    (from onnxruntime-android 1.26.0)
├── armeabi-v7a/
│   ├── libvietasr.so       5.0 MB
│   └── libonnxruntime.so   19 MB
└── x86_64/
    ├── libvietasr.so       5.6 MB
    └── libonnxruntime.so   33 MB
```

## Install (once an `.aar` is published)

```kotlin
// In your app's build.gradle.kts
dependencies {
    implementation("io.vietasr:vietasr:0.1.0")
}
```

## Quickstart (Kotlin)

```kotlin
import io.vietasr.Pipeline

val pipe = Pipeline.preset("transcribe")
val result = pipe.transcribe("/sdcard/audio.wav")
println(result.text)
pipe.close()
```

## Streaming from the mic

```kotlin
import android.media.AudioRecord
import io.vietasr.Pipeline

val pipe = Pipeline.preset("transcribe")
val session = pipe.stream(16000f)

val record = AudioRecord(/* mic source, 16000 Hz, mono 16-bit */)
record.startRecording()
val buffer = ShortArray(1024)

while (recording) {
    val n = record.read(buffer, 0, buffer.size)
    val chunk = buffer.copyOfRange(0, n)
    session.accept(chunk)
    runOnUiThread { textView.text = session.partial().text }
}

val final = session.finalResult().text
session.close()
pipe.close()
```

## Custom pipeline

```kotlin
val pipe = Pipeline.newPipeline()
    .add("vad")
    .add("vietasr")
    .add("punctuation")
    .add("gender")
    .add("emotion")
    .build()

val result = pipe.transcribe("/sdcard/call.wav")
println(result.text)
println(result.toJson())  // raw JSON has all module fields
pipe.close()
```

## Building from source

Requires Android Studio or the standalone command-line tools (NDK, SDK, Gradle).

### Native libs only (no Android SDK needed)

```bash
NDK=/path/to/android-ndk-r26d
cd bindings/android
for ABI in arm64-v8a armeabi-v7a x86_64; do
    mkdir -p build/$ABI && cd build/$ABI
    cmake ../../lib/src/main/cpp \
        -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI=$ABI \
        -DANDROID_PLATFORM=android-24 \
        -DANDROID_STL=c++_shared \
        -DONNXRUNTIME_ROOT=/path/to/extracted/onnxruntime-android-1.26.0.aar
    cmake --build . --parallel
    cd ../..
    cp build/$ABI/libvietasr.so lib/src/main/jniLibs/$ABI/
done
```

The ONNX Runtime Android AAR is at:
```
https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.26.0/onnxruntime-android-1.26.0.aar
```
Unzip it; `jni/<abi>/libonnxruntime.so` are what you need.

### Full AAR build (needs Android SDK + Gradle)

```bash
cd bindings/android
./gradlew :lib:assembleRelease
# Output: lib/build/outputs/aar/lib-release.aar
```

### Demo app

```bash
./gradlew :demo:installDebug
adb shell am start -n io.vietasr.demo/.MainActivity
```

The demo expects a `sample.wav` (16 kHz mono 16-bit PCM) in `demo/src/main/assets/`.

## API

| Class | Method | Notes |
|---|---|---|
| `Pipeline` | `preset(name)` | static |
| `Pipeline` | `newPipeline()` | static |
| `Pipeline` | `add(moduleName)` / `add(moduleName, jsonConfig)` | fluent |
| `Pipeline` | `setBackend(Backend)` | `ONNX` (default) |
| `Pipeline` | `setModelDir(path)` | advanced; load model from a directory instead of the embedded one |
| `Pipeline` | `build()` | finalise; called automatically by `preset(...)` |
| `Pipeline` | `transcribe(wavPath)` | returns `Result` |
| `Pipeline` | `transcribe(short[], sampleRate)` | in-memory PCM |
| `Pipeline` | `stream(sampleRate)` | returns `Session` |
| `Pipeline` | `listModules()` / `listPresets()` | static |
| `Pipeline` | `version()` | static |
| `Pipeline` | `setLogLevel(int)` | static; 0=trace 1=debug 2=info 3=warn 4=error 5=off |
| `Pipeline` | `close()` | implements `AutoCloseable` |
| `Session` | `accept(short[])` / `accept(float[])` | returns true if endpoint reached |
| `Session` | `partial()` / `result()` / `finalResult()` | returns `Result` |
| `Session` | `reset()` / `close()` | — |
| `Result` | `getText()`, `getPartial()`, `isFinal()`, `toJson()` | typed accessors over the JSON envelope |

## Thread safety

- One `Pipeline` can serve many concurrent `Session`s. Each Session has its own state via `Module::Clone()`.
- See [docs/thread-safety.md](../../docs/thread-safety.md).

## Models

The vietasr model is **bundled inside the SDK** — no download, no network, no asset
copying. `model.onnx` is committed to the repo as <50 MB chunks and baked into the
native `libvietasr` shipped in the AAR.
