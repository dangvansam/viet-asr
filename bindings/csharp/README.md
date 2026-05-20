# vietasr — .NET binding

Offline Vietnamese Speech AI SDK for .NET. P/Invoke wrapper over the same C ABI that powers every other vietasr binding. Works on Linux, macOS, and Windows.

## Install

```bash
dotnet add package Vietasr
```

The NuGet package bundles prebuilt native libraries under `runtimes/<rid>/native/`. A `DllImportResolver` ([NativeLoader.cs](Vietasr/NativeLoader.cs)) finds and loads them — including `libonnxruntime` — at runtime.

## Quickstart

```csharp
using Vietasr;

using var pipe = Pipeline.Preset("transcribe");
Result result = pipe.Transcribe("audio.wav");
Console.WriteLine(result.Text);
```

## Streaming

```csharp
using Vietasr;

using var pipe = Pipeline.Preset("transcribe");
using var session = pipe.Stream(16000);

foreach (short[] chunk in MicChunks())
{
    session.Accept(chunk);
    Console.WriteLine(session.Partial().Text);
}
Console.WriteLine("FINAL: " + session.Final().Text);
```

## Custom pipeline

```csharp
using var pipe = Pipeline.Create()
    .Add("vad")
    .Add("vietasr")
    .Add("punctuation")
    .Add("gender")
    .Add("emotion")
    .Build();

Result result = pipe.Transcribe("call.wav");
Console.WriteLine(result.Text);
Console.WriteLine(result.Field("gender"));   // JsonElement?
```

## API

| Type | Member | Notes |
|---|---|---|
| `Pipeline` | `Preset(string)` | static; model bundled in the SDK |
| `Pipeline` | `Create()` | static; empty pipeline |
| `Pipeline` | `Add(module)` / `Add(module, jsonConfig)` | fluent |
| `Pipeline` | `SetBackend(Backend)` | `Auto`, `Onnx`, `CoreML` |
| `Pipeline` | `SetModelDir(path)` | advanced; load model from a directory instead of the embedded one |
| `Pipeline` | `Build()` | finalise; auto-called by `Preset` |
| `Pipeline` | `Transcribe(wavPath)` / `Transcribe(short[], sampleRate)` | returns `Result` |
| `Pipeline` | `Stream(sampleRate)` | returns `Session` |
| `Pipeline` | `ListModules()` / `ListPresets()` / `Version()` | static |
| `Pipeline` | `Dispose()` | implements `IDisposable` |
| `Session` | `Accept(short[])` / `Accept(float[])` | true if endpoint reached |
| `Session` | `Partial()` / `GetResult()` / `Final()` | returns `Result` |
| `Session` | `Reset()` / `Dispose()` | — |
| `Result` | `Text`, `Partial`, `IsFinal`, `Raw` | typed accessors |
| `Result` | `Field(key)` | `JsonElement?` for extra module fields |
| `Result` | `ToJson()` | raw JSON string |

## Build from source

```bash
# stage native libs (after building core/)
cp ../../build-core/libvietasr.so.*          runtimes/linux-x64/native/libvietasr.so
cp ../../build-core/.../libonnxruntime.so.*  runtimes/linux-x64/native/libonnxruntime.so

dotnet build Vietasr.Examples/Vietasr.Examples.csproj -c Release

# batch
dotnet Vietasr.Examples/bin/Release/net8.0/Quickstart.dll audio.wav
# streaming
dotnet Vietasr.Examples/bin/Release/net8.0/Quickstart.dll --stream audio.wav
```

## Pack a NuGet

```bash
dotnet pack Vietasr/Vietasr.csproj -c Release
# -> Vietasr/bin/Release/Vietasr.0.1.0.nupkg
```

## Thread safety

- One `Pipeline` can serve many concurrent `Session`s — each Session has independent state.
- See [docs/thread-safety.md](../../docs/thread-safety.md).

## Models

The vietasr model is **bundled inside the SDK** — no download, no network. `model.onnx`
is committed to the repo as <50 MB chunks and baked into the native `libvietasr`
shipped in the NuGet package.
