# VietASR — Swift (iOS / macOS)

Offline Vietnamese Speech AI for Apple platforms. Built on the same C/C++ core
that powers every other VietASR binding.

- **Platforms:** iOS 13+, macOS 11+ — device + simulator. One build runs on
  those versions and every newer release (through the current iOS / macOS).
- **Self-contained:** the ~66 MB ASR model and ONNX Runtime are baked into the
  framework — no downloads, no network, no API keys.

### Supported OS range

The xcframework is built once and runs across the whole range — Apple
guarantees forward compatibility from the deployment target upward.

| Platform | Minimum | Why that floor |
|----------|---------|----------------|
| iOS / iPadOS | 13.0 | ONNX Runtime 1.20 iOS slice + `std::filesystem` |
| macOS | 11.0 | ONNX Runtime 1.20 macOS slice |

To raise the minimum (e.g. to drop older OSes), rebuild with
`IOS_DEPLOYMENT_TARGET=15.0 MACOS_DEPLOYMENT_TARGET=12.0 bash build-xcframework.sh`
and bump the `platforms:` in `Package.swift` to match.

## Install

Swift Package Manager:

```swift
.package(url: "https://github.com/dangvansam/viet-asr.git", from: "0.1.0")
```

Then add `VietASR` to your target's dependencies and `import VietASR`.

> The model adds **~66 MB** to your app. That is the cost of fully offline,
> zero-config recognition — the same trade-off every VietASR binding makes.

## Quickstart

```swift
import VietASR

let pipeline = try Pipeline.preset("transcribe")

// Batch — a whole clip of 16 kHz, 16-bit mono PCM.
let result = try pipeline.transcribe(pcmSamples, sampleRate: 16_000)
print(result.text)

// Batch — a WAV file.
let fromFile = try pipeline.transcribe(file: "/path/to/audio.wav")
print(fromFile.text)
```

## Streaming

```swift
let pipeline = try Pipeline.preset("transcribe")
let session  = try pipeline.stream(sampleRate: 16_000)

for chunk in micChunks {                 // [Int16] or [Float]
    let status = session.accept(chunk)
    print(session.partialResult.text)
    if status == .final {
        print("segment:", session.finalResult.text)
    }
}
print("FINAL:", session.finalResult.text)
```

## Custom pipeline

```swift
let pipeline = try Pipeline.create()
try pipeline.add("vad").add("vietasr").build()
let result = try pipeline.transcribe(pcmSamples)
```

Discover what is available:

```swift
VietASR.version            // core version string
VietASR.listPresets()      // ["transcribe", "analytics", ...]
VietASR.listModules()      // ["vad", "vietasr", "punctuation", ...]
```

## Building from source

`CVietASR.xcframework` is ~470 MB (the model + ONNX Runtime are baked in) so it
is never committed. Build it once from a git checkout:

```bash
bash bindings/apple/build-xcframework.sh
```

This cross-compiles the C++ core for iOS device, iOS simulator and macOS,
links in ONNX Runtime (from the official `onnxruntime-c` CocoaPods archive),
and assembles `bindings/apple/artifacts/CVietASR.xcframework`. Re-run it after
any change to the C++ core.

Requirements: Xcode, CMake 3.20+.

`Package.swift` resolves the xcframework automatically: when
`artifacts/CVietASR.xcframework` exists (a local checkout that has run the
script) it links it directly; otherwise — a consumer fetching a version tag —
it falls back to the GitHub Release asset. No manual switching.

## Testing

```bash
cd bindings/apple
swift test                                              # runs on the macOS host
xcodebuild test -scheme VietASR \
  -destination 'platform=iOS Simulator,name=iPhone 17'  # runs on a simulator
```

## Releasing

Releases are fully automated by the `release` GitHub Actions workflow. On a
push to the `sdk` branch (or a `v*` tag) its `build-apple` job:

1. builds `CVietASR.xcframework` and computes its checksum;
2. uploads `CVietASR.xcframework.zip` as a GitHub Release asset;
3. stamps the release `url:` + `checksum:` into `Package.swift`
   (`scripts/stamp-apple-package.py`) and commits it so the release tag
   carries a manifest that `.package(url:)` consumers resolve directly.

No manual steps. To cut a release locally instead, run `build-xcframework.sh`,
`swift package compute-checksum artifacts/CVietASR.xcframework.zip`, then
`python3 scripts/stamp-apple-package.py <version> <checksum>` before tagging.
