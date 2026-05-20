import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import vietasr


def transcribe_with_own_pipeline(wav_path: str) -> str:
    pipe = vietasr.Pipeline.preset("transcribe")
    return pipe.transcribe(wav_path).text


def transcribe_with_shared_pipeline(pipe, wav_path: str, lock: threading.Lock) -> str:
    with lock:
        return pipe.transcribe(wav_path).text


def stress_pattern_1(wav_path: str, n_workers: int) -> tuple[set[str], float]:
    started = time.perf_counter()
    results = set()
    errors = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(transcribe_with_own_pipeline, wav_path) for _ in range(n_workers)]
        for fut in as_completed(futures):
            try:
                results.add(fut.result())
            except Exception as exc:
                errors.append(repr(exc))
    elapsed = time.perf_counter() - started
    if errors:
        print(f"  errors: {errors}")
    return results, elapsed


def stress_pattern_2(wav_path: str, n_workers: int) -> tuple[set[str], float]:
    pipe = vietasr.Pipeline.preset("transcribe")
    lock = threading.Lock()
    started = time.perf_counter()
    results = set()
    errors = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(transcribe_with_shared_pipeline, pipe, wav_path, lock)
                   for _ in range(n_workers)]
        for fut in as_completed(futures):
            try:
                results.add(fut.result())
            except Exception as exc:
                errors.append(repr(exc))
    elapsed = time.perf_counter() - started
    if errors:
        print(f"  errors: {errors}")
    return results, elapsed


def stress_pattern_3_unsafe(wav_path: str, n_workers: int) -> tuple[set[str], float]:
    pipe = vietasr.Pipeline.preset("transcribe")
    started = time.perf_counter()
    results = set()
    errors = []
    def worker():
        try:
            return pipe.transcribe(wav_path).text
        except Exception as exc:
            return f"<error: {exc!r}>"
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(worker) for _ in range(n_workers)]
        for fut in as_completed(futures):
            results.add(fut.result())
    elapsed = time.perf_counter() - started
    return results, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    print(f"Stress test: {args.workers} threads transcribing the same WAV.\n")

    print("== Pattern 1 — one Pipeline per worker (parallel-safe) ==")
    p1_results, p1_time = stress_pattern_1(args.wav_path, args.workers)
    print(f"  unique transcripts: {len(p1_results)}")
    print(f"  wall time: {p1_time:.2f}s")
    if len(p1_results) == 1:
        print(f"  OK — all {args.workers} workers produced identical output")
    else:
        print(f"  FAIL — {len(p1_results)} different transcripts:")
        for t in p1_results:
            print(f"    {t[:80]!r}")
        sys.exit(1)

    print()
    print("== Pattern 2 — one Pipeline, lock around access (serialised) ==")
    p2_results, p2_time = stress_pattern_2(args.wav_path, args.workers)
    print(f"  unique transcripts: {len(p2_results)}")
    print(f"  wall time: {p2_time:.2f}s")
    if len(p2_results) == 1 and p2_results == p1_results:
        print(f"  OK — same transcript as pattern 1")
    else:
        print(f"  FAIL — pattern 2 diverged")
        sys.exit(1)

    print()
    print(f"Pattern 1 vs Pattern 2 speedup: {p2_time/p1_time:.2f}x")

    print()
    print("== Pattern 3 — UNSAFE: shared Pipeline, no lock (documented as racy) ==")
    p3_results, p3_time = stress_pattern_3_unsafe(args.wav_path, args.workers)
    print(f"  unique transcripts: {len(p3_results)}")
    print(f"  wall time: {p3_time:.2f}s")
    if len(p3_results) == 1 and p3_results == p1_results:
        print("  OK by luck (single utterance, low contention)")
    else:
        print("  RACE — Pattern 3 produced divergent outputs as documented:")
        for t in p3_results:
            print(f"    {t[:120]!r}")


if __name__ == "__main__":
    main()
