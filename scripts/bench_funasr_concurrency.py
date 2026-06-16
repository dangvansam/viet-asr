import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List

import requests
import soundfile as sf


@dataclass
class LevelResult:
    concurrency: int
    clips: int
    wall_seconds: float
    audio_seconds: float
    clips_per_sec: float
    rtfx: float
    failures: int


class ConcurrencyBenchmark:
    def __init__(self, url: str, paths: List[str], language: str, model: str = "fun-asr-nano"):
        self.url = url
        self.paths = paths
        self.language = language
        self.model = model
        self.audio_seconds = sum(self.duration(p) for p in paths)
        self.texts_by_level: Dict[int, Dict[str, str]] = {}

    def duration(self, path: str) -> float:
        info = sf.info(path)
        return info.frames / info.samplerate

    def transcribe(self, path: str) -> str:
        with open(path, "rb") as handle:
            files = {"file": (os.path.basename(path), handle, "audio/wav")}
            data = {"model": self.model, "language": self.language, "response_format": "json"}
            response = requests.post(self.url, files=files, data=data, timeout=300)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return payload.get("text", "")
        return str(payload)

    def warmup(self) -> None:
        self.transcribe(self.paths[0])

    def run_level(self, concurrency: int) -> LevelResult:
        texts: Dict[str, str] = {}
        failures = 0
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(self.transcribe, p): p for p in self.paths}
            for future, path in [(f, futures[f]) for f in futures]:
                try:
                    texts[path] = future.result()
                except Exception as exc:
                    failures += 1
                    texts[path] = f"<ERROR: {exc}>"
        wall = time.perf_counter() - start
        self.texts_by_level[concurrency] = texts
        clips_per_sec = len(self.paths) / wall if wall else 0.0
        rtfx = self.audio_seconds / wall if wall else 0.0
        return LevelResult(
            concurrency=concurrency,
            clips=len(self.paths),
            wall_seconds=wall,
            audio_seconds=self.audio_seconds,
            clips_per_sec=clips_per_sec,
            rtfx=rtfx,
            failures=failures,
        )

    def text_mismatches(self, base: int, other: int) -> int:
        a = self.texts_by_level.get(base, {})
        b = self.texts_by_level.get(other, {})
        return sum(1 for k in a if a.get(k) != b.get(k))


def main() -> None:
    parser = argparse.ArgumentParser(description="FunASR concurrency / batch throughput benchmark")
    parser.add_argument("--url", type=str, default="http://localhost:9102/v1/audio/transcriptions")
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--concurrency", type=str, default="1,8,16")
    parser.add_argument("--language", type=str, default="vi")
    parser.add_argument("--model", type=str, default="fun-asr-nano")
    parser.add_argument("--show-mismatches", action="store_true")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "*.wav")))[: args.n]
    if not paths:
        raise SystemExit(f"no wavs in {args.dir}")

    bench = ConcurrencyBenchmark(args.url, paths, args.language, args.model)
    print(f"clips={len(paths)} audio={bench.audio_seconds:.1f}s url={args.url}")
    bench.warmup()

    levels = [int(c) for c in args.concurrency.split(",") if c.strip()]
    results = [bench.run_level(c) for c in levels]

    print("\n" + "=" * 78)
    print(f"{'CONCURRENCY':<12}{'WALL(s)':<10}{'CLIPS/s':<10}{'RTFx':<10}{'FAIL':<6}")
    print("-" * 78)
    for r in results:
        print(f"{r.concurrency:<12}{r.wall_seconds:<10.2f}{r.clips_per_sec:<10.2f}{r.rtfx:<10.1f}{r.failures:<6}")
    print("-" * 78)
    base = levels[0]
    for c in levels[1:]:
        print(f"text mismatches vs concurrency={base}: {bench.text_mismatches(base, c)} / {len(paths)}")
    print("=" * 78)

    if args.show_mismatches and len(levels) > 1:
        top = levels[-1]
        a = bench.texts_by_level.get(base, {})
        b = bench.texts_by_level.get(top, {})
        print(f"\nMISMATCHES concurrency={base} vs {top}")
        for path in a:
            if a.get(path) != b.get(path):
                print(f"\n[{os.path.basename(path)}]")
                print(f"  c={base}: {a.get(path)!r}")
                print(f"  c={top}: {b.get(path)!r}")


if __name__ == "__main__":
    main()
