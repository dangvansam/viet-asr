# vietasr — Go binding

Offline Vietnamese Speech AI SDK for Go. cgo wrapper over the same C ABI that powers every other vietasr binding.

## Install

```bash
go get github.com/dangvansam/viet-asr/bindings/go
```

The package vendors a prebuilt `libvietasr.{so,dylib,dll}` + `libonnxruntime.{so,dylib,dll}` under `_native/`. The cgo `-rpath` ensures the binary finds them at runtime.

## Quickstart

```go
package main

import (
    "fmt"
    vietasr "github.com/dangvansam/viet-asr/bindings/go"
)

func main() {
    pipe, err := vietasr.PipelinePreset("transcribe")
    if err != nil {
        panic(err)
    }
    defer pipe.Close()

    result, err := pipe.TranscribeFile("audio.wav")
    if err != nil {
        panic(err)
    }
    fmt.Println(result.Text)
}
```

## Streaming

```go
session, err := pipe.Stream(16000)
defer session.Close()

for _, chunk := range micChunks {
    session.Accept(chunk)             // []int16
    fmt.Print("\r", session.Partial().Text)
}
fmt.Println("\n", session.Final().Text)
```

## Custom pipeline

```go
pipe := vietasr.NewPipeline()
defer pipe.Close()

pipe.Add("vad", nil)
pipe.Add("vietasr", nil)
pipe.Add("gender", nil)
pipe.Add("emotion", nil)
pipe.Build()

result, _ := pipe.TranscribeFile("call.wav")
fmt.Println(result.Text)
fmt.Println(result.Field("gender"))
```

## API

| Type/Func | Returns |
|---|---|
| `PipelinePreset(name string)` | `(*Pipeline, error)` |
| `NewPipeline()` | `*Pipeline` |
| `(*Pipeline).Add(name string, config map[string]any) error` | error |
| `(*Pipeline).SetBackend(Backend) error` | error |
| `(*Pipeline).SetModelDir(path string) error` | error |
| `(*Pipeline).Build() error` | error |
| `(*Pipeline).TranscribeFile(path string)` | `(Result, error)` |
| `(*Pipeline).TranscribeBuffer(pcm []int16, sr float32)` | `(Result, error)` |
| `(*Pipeline).Stream(sr float32)` | `(*Session, error)` |
| `(*Pipeline).Close()` | — |
| `(*Session).Accept(pcm []int16) bool` | endpoint reached |
| `(*Session).AcceptFloat(pcm []float32) bool` | endpoint reached |
| `(*Session).Partial() Result` | — |
| `(*Session).Result() Result` | — |
| `(*Session).Final() Result` | — |
| `(*Session).Reset()` / `Close()` | — |
| `Result.Text`, `.Partial`, `.IsFinal`, `.Segments`, `.Speakers` | typed |
| `Result.Field(key string) any` | extra JSON fields |
| `Result.Raw` | raw JSON string |
| `ListModules()`, `ListPresets()`, `Version()`, `SetLogLevel(level)` | — |

## Examples

- [examples/quickstart](examples/quickstart/main.go) — batch transcribe one file
- [examples/streaming](examples/streaming/main.go) — chunked feed + live partials
- [examples/custom](examples/custom/main.go) — composable analytics pipeline

## Thread safety

- One `Pipeline` can serve many concurrent `Session`s — each Session has its own state.
- For pure parallelism on streaming, share one Pipeline across many Sessions.
- For batch in workers, one Pipeline per worker still works (slightly more memory).
- See [docs/thread-safety.md](../../docs/thread-safety.md) for the full guide.

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
