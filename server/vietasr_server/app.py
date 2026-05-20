"""FastAPI application: OpenAI-compatible REST + WebSocket streaming."""
from __future__ import annotations

import json

from fastapi import (FastAPI, File, Form, HTTPException, UploadFile,
                     WebSocket, WebSocketDisconnect)
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool

from vietasr_server import __version__
from vietasr_server.asr import MODEL_ID, AsrEngine
from vietasr_server.audio import AudioDecodeError, decode_audio, pcm_from_bytes
from vietasr_server.ui import INDEX_HTML

_CONTROL_ACTIONS = {"eof", "done", "stop", "close"}


def create_app(engine: AsrEngine) -> FastAPI:
    app = FastAPI(
        title="VietASR Server",
        version=__version__,
        description="Offline Vietnamese speech-to-text — OpenAI-compatible REST "
                    "for batch + WebSocket for streaming.",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": MODEL_ID, "pipeline": engine.description}

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return INDEX_HTML

    @app.get("/v1/models")
    def list_models() -> dict:
        return {
            "object": "list",
            "data": [{
                "id": MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "vietasr",
            }],
        }

    # ---- OpenAI-compatible non-streaming transcription -------------------
    @app.post("/v1/audio/transcriptions")
    async def create_transcription(
        file: UploadFile = File(...),
        model: str = Form(MODEL_ID),
        response_format: str = Form("json"),
        language: str = Form(None),
        temperature: float = Form(None),
    ):
        del model, temperature  # single model; sampling not applicable
        data = await file.read()
        try:
            pcm, sample_rate = decode_audio(data)
        except AudioDecodeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        result = await run_in_threadpool(engine.transcribe, pcm, float(sample_rate))
        text = result.text
        fmt = (response_format or "json").lower()

        if fmt == "text":
            return PlainTextResponse(text + "\n")
        if fmt == "verbose_json":
            return JSONResponse({
                "task": "transcribe",
                "language": language or "vietnamese",
                "duration": round(len(pcm) / sample_rate, 3),
                "text": text,
                "segments": result.segments,
            })
        return JSONResponse({"text": text})

    # ---- WebSocket streaming transcription ------------------------------
    @app.websocket("/v1/stream")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        try:
            sample_rate = float(ws.query_params.get("sample_rate", "16000"))
        except (TypeError, ValueError):
            sample_rate = 16000.0
        encoding = ws.query_params.get("encoding", "pcm_s16le")

        session = await run_in_threadpool(engine.new_session, sample_rate)
        await ws.send_json({"type": "ready", "sample_rate": sample_rate,
                            "encoding": encoding})
        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                chunk = message.get("bytes")
                text = message.get("text")

                if chunk:
                    try:
                        pcm = pcm_from_bytes(chunk, encoding)
                    except AudioDecodeError as exc:
                        await ws.send_json({"type": "error", "message": str(exc)})
                        continue
                    await run_in_threadpool(session.accept, pcm)
                    partial = await run_in_threadpool(session.partial)
                    await ws.send_json({"type": "partial", "text": partial.text,
                                        "is_final": False})
                elif text:
                    try:
                        action = json.loads(text).get("type")
                    except (json.JSONDecodeError, AttributeError):
                        action = None
                    if action in _CONTROL_ACTIONS:
                        final = await run_in_threadpool(session.final)
                        await ws.send_json({"type": "final", "text": final.text,
                                            "is_final": True})
                        break
        except WebSocketDisconnect:
            pass
        finally:
            session.close()
            try:
                await ws.close()
            except RuntimeError:
                pass

    return app
