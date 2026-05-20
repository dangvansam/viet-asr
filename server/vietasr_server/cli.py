"""`vietasr-server` — one command to launch the ASR server."""
from __future__ import annotations

import argparse

from vietasr_server import __version__


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="vietasr-server",
        description="Launch the VietASR server: OpenAI-compatible REST "
                    "(/v1/audio/transcriptions) + WebSocket streaming (/v1/stream).",
    )
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="bind port (default: 8000)")
    parser.add_argument("--preset", default="transcribe",
                        help="pipeline preset (default: transcribe)")
    parser.add_argument("--module", action="append", default=[], metavar="NAME",
                        help="build a custom pipeline from modules (overrides --preset)")
    parser.add_argument("--log-level", default="info",
                        choices=["critical", "error", "warning", "info", "debug", "trace"],
                        help="uvicorn log level (default: info)")
    parser.add_argument("--version", action="version", version=f"vietasr-server {__version__}")
    args = parser.parse_args()

    # Import lazily so --help / --version work without loading the native library.
    import uvicorn

    from vietasr_server.app import create_app
    from vietasr_server.asr import AsrEngine

    print(f"vietasr-server {__version__} — loading model ...", flush=True)
    engine = AsrEngine(preset=args.preset, modules=args.module or None)
    app = create_app(engine)

    print(f"  pipeline : {engine.description}", flush=True)
    print(f"  REST     : http://{args.host}:{args.port}/v1/audio/transcriptions", flush=True)
    print(f"  WebSocket: ws://{args.host}:{args.port}/v1/stream", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
