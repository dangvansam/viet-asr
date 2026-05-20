"""Minimal streaming client for the VietASR WebSocket endpoint.

Usage:  python ws_client.py audio.wav [ws://host:port/v1/stream]

Streams a 16-bit PCM WAV in real-time-ish chunks and prints partial
transcripts as they arrive, then the final transcript.
"""
import asyncio
import json
import struct
import sys
import wave


def read_wav_s16(path: str):
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit("this demo expects a 16-bit PCM WAV")
        channels = w.getnchannels()
        sample_rate = w.getframerate()
        frames = w.readframes(w.getnframes())
    if channels > 1:  # down-mix to mono
        samples = struct.unpack(f"<{len(frames) // 2}h", frames)
        mono = [sum(samples[i:i + channels]) // channels
                for i in range(0, len(samples), channels)]
        frames = struct.pack(f"<{len(mono)}h", *mono)
    return frames, sample_rate


async def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python ws_client.py audio.wav [ws-url]")
    try:
        import websockets
    except ImportError:
        raise SystemExit("pip install websockets")

    pcm, sample_rate = read_wav_s16(sys.argv[1])
    base = sys.argv[2] if len(sys.argv) > 2 else "ws://localhost:8000/v1/stream"
    url = f"{base}?sample_rate={sample_rate}&encoding=pcm_s16le"

    chunk = sample_rate // 5 * 2  # ~200 ms of 16-bit mono per message

    async with websockets.connect(url) as ws:
        print(json.loads(await ws.recv()))  # {"type":"ready",...}

        async def receiver():
            async for raw in ws:
                msg = json.loads(raw)
                if msg["type"] == "partial":
                    print(f"  ~ {msg['text']}")
                elif msg["type"] == "final":
                    print(f"\nFINAL: {msg['text']}")
                    return
                elif msg["type"] == "error":
                    print(f"  ! {msg['message']}")

        recv_task = asyncio.create_task(receiver())
        for i in range(0, len(pcm), chunk):
            await ws.send(pcm[i:i + chunk])
            await asyncio.sleep(0.05)
        await ws.send(json.dumps({"type": "eof"}))
        await recv_task


if __name__ == "__main__":
    asyncio.run(main())
