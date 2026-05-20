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

### CLI (Python, any OS)

```bash
pip install ./bindings/python ./server
vietasr-server                      # -> http://0.0.0.0:8000
```

`vietasr-server --host 127.0.0.1 --port 9000 --log-level debug` to customize.

## Test console

Open **`http://localhost:8000/`** in a browser — a built-in page to try both
endpoints: upload a file (batch), or record from the mic / stream a file
(WebSocket) and watch partial transcripts live.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/` | Browser test console |
| `POST` | `/v1/audio/transcriptions` | Batch transcription (OpenAI-compatible) |
| `WS`   | `/v1/stream` | Streaming transcription |
| `GET`  | `/v1/models` | List models (OpenAI-compatible) |
| `GET`  | `/health` | Liveness probe |

## Batch — OpenAI-compatible

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F response_format=json
# {"text":"xin chào, bạn nghe rõ không"}
```

`response_format` accepts `json` (default), `text`, `verbose_json`.
WAV (16-bit PCM) works out of the box; other formats (mp3, m4a, flac, …)
need `ffmpeg` on PATH (already in the Docker image).

Works with the OpenAI SDK by pointing `base_url` at the server:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
with open("audio.wav", "rb") as f:
    print(client.audio.transcriptions.create(model="vietasr", file=f).text)
```

## Streaming — WebSocket

Connect to `ws://localhost:8000/v1/stream?sample_rate=16000&encoding=pcm_s16le`.

- **Client → server:** binary messages of raw PCM
  (`pcm_s16le` default, or `pcm_f32le`).
- **Server → client:** JSON — `{"type":"ready",...}` once, then
  `{"type":"partial","text":...}` per chunk.
- Send a text message `{"type":"eof"}` to finish; the server replies with
  `{"type":"final","text":...,"is_final":true}` and closes.

A ready-made client is in [examples/ws_client.py](examples/ws_client.py):

```bash
python server/examples/ws_client.py audio.wav
```

## CLI options

```
vietasr-server [--host H] [--port P] [--preset NAME] [--module NAME ...]
               [--log-level LEVEL]
```

- `--preset` — pipeline preset (default `transcribe`).
- `--module` — build a custom pipeline (repeatable; overrides `--preset`).
