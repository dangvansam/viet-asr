# vietasr-server

Offline Vietnamese speech-to-text as a server:

- **`POST /v1/audio/transcriptions`** — OpenAI-compatible batch transcription.
- **`WebSocket /v1/stream`** — low-latency streaming transcription.

One process serves both. The model is embedded in the SDK — no downloads, no
API keys, no cloud.

## Run it — one command

### Docker Compose (any OS)

```bash
docker compose up           # from the repo root; builds the image on first run
```

Server is on `http://localhost:8000`. Override the host port with
`VIETASR_PORT=9000 docker compose up`.

This default image is small and multi-arch (amd64 + arm64) — the right choice
for **edge / CPU-only** devices. It runs on ONNX Runtime's CPU provider, which
is the best runtime ONNX Runtime's standard prebuilts expose on those targets.

### GPU server (NVIDIA CUDA)

For a server with an NVIDIA GPU, use the CUDA image — it bundles the ONNX
Runtime CUDA execution provider and auto-detects the GPU at startup (falling
back to CPU if none is visible). It needs the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
on the host.

```bash
docker compose --profile gpu up server-gpu        # build + run locally
```

Or directly, or from Docker Hub (`dangvansam/viet-asr:gpu`):

```bash
docker build -f server/Dockerfile.gpu -t viet-asr:gpu .
docker run --gpus all -p 8000:8000 viet-asr:gpu
```

The startup log prints the active runtime — `runtime : CUDAExecutionProvider`
when the GPU is engaged, `CPUExecutionProvider` if it fell back.

### CLI (Python, any OS)

```bash
pip install ./bindings/python ./server
vietasr-server                      # -> http://0.0.0.0:8000
```

`vietasr-server --host 127.0.0.1 --port 9000 --log-level debug` to customize.

## Test console

Open **`http://localhost:8000/`** in a browser — a built-in page to exercise
every feature: pick a pipeline (a preset, or a custom module set to test one
model on its own), then run it in **batch** (upload a file) or **streaming**
(record from the mic / stream a file) mode. Each run shows the full module
result JSON — `text`, `voiced_regions`, `speech`, `itn`, `segments`, … —
so all module outputs are visible.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/` | Browser test console |
| `POST` | `/v1/audio/transcriptions` | Batch transcription (OpenAI-compatible) |
| `WS`   | `/v1/stream` | Streaming transcription |
| `GET`  | `/v1/pipelines` | List selectable presets + modules |
| `GET`  | `/v1/models` | List models (OpenAI-compatible) |
| `GET`  | `/health` | Liveness probe |

## Selecting a pipeline

Every request can pick the pipeline to run — a built-in **preset**
(`transcribe`, `transcribe-rich`, `stream`, `vad`, `meeting`, `analytics`) or a
**custom set of modules** (`vad`, `vietasr`, `itn`, `punctuation`, …) to
exercise one model on its own. `GET /v1/pipelines` lists what is available.
Engines are built on first use and cached, so the first request for a new
pipeline is slower. With no selection, the server default (`--preset` /
`--module`) runs.

The **`stream`** preset is the recommended pipeline for streaming — ASR-only,
no separate VAD model. The CTC endpointer segments on silence; the ASR emits
`voiced_regions` + `speech` (from CTC-blank), a per-sentence `segments[]` array
(`text` + `start`/`end`), and ITN-normalizes each sentence the moment it ends —
all while the encoder cache keeps streaming. `text` is the running aggregate.

## Batch — OpenAI-compatible

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F response_format=json
# {"text":"xin chào, bạn nghe rõ không"}
```

`response_format` accepts `json` (default), `text`, `verbose_json` — the last
returns the full module payload (`voiced_regions`, `speech`, `itn`, `segments`,
…). WAV (16-bit PCM) works out of the box; other formats (mp3, m4a, flac, …)
need `ffmpeg` on PATH (already in the Docker image).

Add `-F pipeline=NAME` to run a preset, or `-F modules=vad,vietasr,itn` to run
a custom module set:

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav -F response_format=verbose_json -F pipeline=vad
```

**Long audio** — `-F vad_segment=auto|on|off` (default `auto`). When it applies,
the server runs a neural VAD first, cuts the audio into ≤60 s pieces at silence
boundaries, transcribes each, and merges — skipping silence and avoiding the
encoder's ~3.6 min length limit. `auto` segments clips longer than 3 minutes.

Works with the OpenAI SDK by pointing `base_url` at the server:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
with open("audio.wav", "rb") as f:
    print(client.audio.transcriptions.create(model="vietasr", file=f).text)
```

## Streaming — WebSocket

Connect to `ws://localhost:8000/v1/stream?sample_rate=16000&encoding=pcm_s16le`.
Add `&pipeline=NAME` or `&modules=vad,vietasr` to pick the pipeline.

- **Client → server:** binary messages of raw PCM
  (`pcm_s16le` default, or `pcm_f32le`).
- **Server → client:** JSON — `{"type":"ready",...}` once, then
  `{"type":"partial","text":...,"result":{...}}` per chunk. `result` carries
  the full live module payload.
- Send a text message `{"type":"eof"}` to finish; the server replies with
  `{"type":"final","text":...,"result":{...},"is_final":true}` and closes.

A ready-made client is in [examples/ws_client.py](examples/ws_client.py):

```bash
python server/examples/ws_client.py audio.wav
```

## CLI options

```
vietasr-server [--host H] [--port P] [--preset NAME] [--module NAME ...]
               [--log-level LEVEL]
```

- `--preset` — the **default** pipeline preset (default `transcribe`).
- `--module` — default custom pipeline (repeatable; overrides `--preset`).

Both set the server default only — any request may still select any preset or
module set via the `pipeline` / `modules` parameters.
