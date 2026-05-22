"""FastAPI application: OpenAI-compatible REST + WebSocket streaming.

Every request may pick the pipeline to run: a built-in preset, or a custom set
of modules (to exercise a single model on its own). The pipeline is selected
with the `pipeline` / `modules` parameters; engines are built on demand and
cached by the EngineRegistry.
"""
from __future__ import annotations

import json
from typing import List, Optional

from fastapi import (FastAPI, File, Form, HTTPException, UploadFile,
                     WebSocket, WebSocketDisconnect)
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool

from vietasr_server import __version__
from vietasr_server.asr import (MODEL_ID, AsrEngine, EngineRegistry,
                                available_pipelines)
from vietasr_server.audio import AudioDecodeError, decode_audio, pcm_from_bytes
from vietasr_server.segment import should_segment, transcribe_segmented
from vietasr_server.ui import INDEX_HTML

_CONTROL_ACTIONS = {"eof", "done", "stop", "close"}


def _split_modules(modules: Optional[str]) -> Optional[List[str]]:
    """Parse a comma-separated module list; None / empty -> None."""
    if not modules:
        return None
    names = [m.strip() for m in modules.split(",") if m.strip()]
    return names or None


def create_app(registry: EngineRegistry) -> FastAPI:
    app = FastAPI(
        title="VietASR Server",
        version=__version__,
        description="Offline Vietnamese speech AI — OpenAI-compatible REST for "
                    "batch + WebSocket for streaming. Any preset or custom "
                    "module set can be selected per request.",
    )

    def resolve_engine(pipeline: Optional[str],
                       modules: Optional[str]) -> AsrEngine:
        """Build/fetch the engine for a request, or raise HTTP 400."""
        try:
            return registry.resolve(preset=pipeline,
                                    modules=_split_modules(modules))
        except Exception as exc:  # PipelineError / unknown preset or module
            raise HTTPException(status_code=400,
                                detail=f"cannot build pipeline: {exc}")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": MODEL_ID,
                "pipeline": registry.default.description}

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

    # ---- Pipelines the test console / API clients can choose ------------
    @app.get("/v1/pipelines")
    def list_pipelines() -> dict:
        info = available_pipelines()
        info["default"] = list(registry.default.spec)
        return info

    # ---- OpenAI-compatible non-streaming transcription -----------------
    @app.post("/v1/audio/transcriptions")
    async def create_transcription(
        file: UploadFile = File(...),
        model: str = Form(MODEL_ID),
        response_format: str = Form("json"),
        language: str = Form(None),
        temperature: float = Form(None),
        pipeline: str = Form(None),
        modules: str = Form(None),
        vad_segment: str = Form("auto"),
    ):
        del model, temperature  # single model; sampling not applicable
        engine = resolve_engine(pipeline, modules)

        data = await file.read()
        try:
            pcm, sample_rate = decode_audio(data)
        except AudioDecodeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        # Long clips: VAD-segment first (cut <=60 s at silence, ASR per chunk).
        duration_s = len(pcm) / sample_rate
        segmented = should_segment(vad_segment, duration_s, engine)
        if segmented:
            payload = await run_in_threadpool(
                transcribe_segmented, registry, engine, pcm, float(sample_rate))
        else:
            result = await run_in_threadpool(
                engine.transcribe, pcm, float(sample_rate))
            payload = result.payload

        text = payload.get("text", "")
        fmt = (response_format or "json").lower()

        if fmt == "text":
            return PlainTextResponse(text + "\n")
        if fmt == "verbose_json":
            # The full module payload — text, segments, voiced_regions,
            # speech, itn, ... — plus how it ran, so the console sees everything.
            return JSONResponse({
                "task": "transcribe",
                "language": language or "vietnamese",
                "duration": round(duration_s, 3),
                "pipeline": engine.description,
                "segmented": segmented,
                **payload,
            })
        return JSONResponse({"text": text})

    # ---- WebSocket streaming transcription -----------------------------
    @app.websocket("/v1/stream")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        try:
            sample_rate = float(ws.query_params.get("sample_rate", "16000"))
        except (TypeError, ValueError):
            sample_rate = 16000.0
        encoding = ws.query_params.get("encoding", "pcm_s16le")

        try:
            engine = registry.resolve(
                preset=ws.query_params.get("pipeline"),
                modules=_split_modules(ws.query_params.get("modules")))
        except Exception as exc:
            await ws.send_json({"type": "error",
                                "message": f"cannot build pipeline: {exc}"})
            await ws.close()
            return

        session = await run_in_threadpool(engine.new_session, sample_rate)
        await ws.send_json({"type": "ready", "sample_rate": sample_rate,
                            "encoding": encoding,
                            "pipeline": engine.description})
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
                        await ws.send_json({"type": "error",
                                            "message": str(exc)})
                        continue
                    await run_in_threadpool(session.accept, pcm)
                    partial = await run_in_threadpool(session.partial)
                    await ws.send_json({"type": "partial",
                                        "text": partial.text,
                                        "result": partial.payload,
                                        "is_final": False})
                elif text:
                    try:
                        action = json.loads(text).get("type")
                    except (json.JSONDecodeError, AttributeError):
                        action = None
                    if action in _CONTROL_ACTIONS:
                        final = await run_in_threadpool(session.final)
                        await ws.send_json({"type": "final",
                                            "text": final.text,
                                            "result": final.payload,
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
