# Thread Safety

This document is the authoritative guide for using `vietasr` across threads. Read it before writing any code that holds a `Pipeline` or `Session` reference from more than one thread.

## TL;DR

- **One `Pipeline` can serve N concurrent streaming `Session`s correctly.** Each Session clones the Pipeline's module templates at construction, so it has its own encoder caches, beam state, endpoint counters, and segment buffer. Sessions share only the read-only Engine and ModelManager.
- **Batch transcription (`Pipeline::TranscribeFile/Buffer`) is serialised under an internal mutex.** Concurrent batch calls on a shared Pipeline don't crash and produce correct results; throughput is bounded to one transcription at a time.
- **For maximum throughput on concurrent batches, use one Pipeline per worker thread.**
- **For maximum throughput on concurrent streams, share one Pipeline across many Sessions.**
- **Global singletons (`Logger`, `ModuleRegistry`, `PresetRegistry`, `ModelManager`) are thread-safe.**
- **The C ABI returns `thread_local` strings.** Each thread sees its own buffer; the lifetime of any returned `const char*` is "until the next call into the SDK on the same thread."

## Audit results

| Component | Verdict | Notes |
|---|---|---|
| `Logger` ([logger.cc](../core/src/utils/logger.cc)) | safe | atomic level, mutex on sink, impl_ now ctor-initialised |
| `ModuleRegistry`, `PresetRegistry` | safe | mutex on map, impl_ now ctor-initialised |
| `ModelManager` ([utils/model_manager.cc](../core/src/utils/model_manager.cc)) | safe | mutex on cache_dir, bundles; downloads go to a unique temp file and are renamed into place, so concurrent first runs (threads or processes) never see a partial model |
| `ResultBuilder` ([result_builder.cc](../core/src/result_builder.cc)) | safe within one instance | mutex around all mutations |
| C ABI strings (`tls_result`, `tls_error`) | per-thread | `thread_local std::string`. Pointer valid until next call on same thread. |
| `OnnxEngine::Run` | safe to call concurrently on same instance | `Ort::Session::Run` is thread-safe per ORT docs; the wrapper's per-call scratch is now stack-local |
| `HttpClient` | safe | libcurl: `curl_global_init` runs once via `std::call_once`, each call uses its own easy handle; WinHTTP: each call opens its own session handles |
| `Pipeline::TranscribeFile/Buffer` | safe under `drive_mutex` (serialised) | calls from N threads serialise; no crash; correct |
| `Session::AcceptWaveform*` / `Partial` / `Result` / `Final` | safe + parallel | Sessions hold cloned modules + per-Session mutex; N Sessions on one Pipeline scale linearly |
| `Pipeline::Build()` | not re-entrant | call once on a fresh Pipeline; not safe to call from multiple threads |
| Module per-instance state (`VietAsrModule::recognizer_`, `segmenter_`, `segments_`, `partial_`) | per-Session via Clone() | each Session has its own VietAsrModule with fresh streaming state; the loaded `TransducerModel` and vocabulary are shared read-only |

## The Pipeline / Session relationship

`Pipeline` owns **template** module objects, initialised once during `Build()` (model load, vocab load). When you call `Pipeline::NewSession()`, the Session calls `Clone()` on each template — the clone reuses the heavy refs (loaded model, vocabulary) but starts with fresh streaming state (own fbank, beam search, VAD segmenter, sentence buffer and partial text).

```cpp
Pipeline pipe;                       // owns templates (engine, vocab loaded once)
auto s1 = pipe.NewSession(16000);    // clones each module: own caches, own beam
auto s2 = pipe.NewSession(16000);    // clones again: independent state from s1

// Thread A:                          Thread B:
s1->AcceptWaveformF32(...);           s2->AcceptWaveformF32(...);
//   ^ mutates s1's modules only      //   ^ mutates s2's modules only
```

The ONNX model (~78 MB int8) stays in process memory exactly once, shared by all Sessions. The per-stream state — buffered audio of the current utterance (≤ 38 s), VAD state and a few beam hypotheses — is per-Session and small.

### Recommended usage patterns

**Pattern A — concurrent streams (recommended for live captioning / server workloads)**

One Pipeline, N Sessions. Engine loaded once; Sessions scale linearly.

```python
pipe = vietasr.Pipeline.preset("transcribe")

def handle_connection(audio_stream):
    with pipe.stream(sample_rate=16000) as session:
        for chunk in audio_stream:
            session.accept(chunk)
            yield session.partial().text
        yield session.final().text
```

**Pattern B — concurrent batch files**

One Pipeline per worker thread. Each worker can saturate one CPU core's worth of encoder work.

```python
def worker(audio_files):
    pipe = vietasr.Pipeline.preset("transcribe")
    for path in audio_files:
        yield pipe.transcribe(path).text
```

Or with a pool:

```python
from queue import Queue
pool = Queue()
for _ in range(8):
    pool.put(vietasr.Pipeline.preset("transcribe"))

def transcribe(path):
    pipe = pool.get()
    try:
        return pipe.transcribe(path).text
    finally:
        pool.put(pipe)
```

## Memory ordering / dangling pointer caveats

- `vietasr_partial_result()` / `vietasr_result()` / `vietasr_final_result()` / `vietasr_transcribe_*` return a `const char*` into a thread-local buffer. **Copy it** if you need to retain it past the next SDK call on the same thread.
- `vietasr_last_error()` lives in the same thread-local buffer scheme. Read it immediately after a failing call.
- `vietasr_list_modules()` / `vietasr_list_presets()` / `vietasr_default_cache_dir()` / `vietasr_version()` all return strings with the same per-thread lifetime semantics.

## Fixes applied in this audit

1. **`OnnxEngine::scratch_int64_` moved from a member to a stack-local in `Run()`.** Without this, two threads calling `engine_->Run()` on the same engine instance would corrupt each other's int64 input scratch buffers. The fix is what enables the "one Pipeline shared by multiple Sessions on the engine layer" pattern referenced in the Roadmap.
2. **Singleton lazy `impl_` init replaced with constructor init** in `Logger`, `ModuleRegistry`, `PresetRegistry`. The previous `if (!instance.impl_) impl_ = make_unique<Impl>()` was a TOCTOU race between concurrent `Instance()` calls during static init. Now `impl_` is constructed inside the constructor, which the C++11 magic-statics rule guarantees runs exactly once.
3. **`curl_global_init` now runs once via `std::call_once`** instead of once per `HttpClient` instance.
4. **Added `Pipeline::Impl::drive_mutex`** guarding `TranscribeFile` and `TranscribeBuffer`. Before this, concurrent batch calls reliably triggered `double free or corruption (out)` (confirmed by [`thread_stress.py`](../bindings/python/examples/thread_stress.py) Pattern 3). After the fix, the same stress test produces zero crashes across 4 worker threads.
5. **`Module::Clone()` + per-Session module instances.** Each `Session` now clones the Pipeline's template modules at construction, so concurrent Sessions on a shared Pipeline have independent streaming state (fbank, beam, VAD segmenter, sentence buffer). The shared `drive_mutex` is no longer needed on the streaming path — each Session owns its own mutex. Confirmed by [`multi_session_stress.py`](../bindings/python/examples/multi_session_stress.py): 4 concurrent streaming Sessions on one Pipeline produce 4 byte-identical correct Vietnamese transcripts.

## Stress test results

**4 concurrent streaming Sessions on one shared Pipeline** ([multi_session_stress.py](../bindings/python/examples/multi_session_stress.py)):
```
unique transcripts: 1
  worker 0: Nửa vòng trái đất hơn bảy năm....
  worker 1: Nửa vòng trái đất hơn bảy năm....
  worker 2: Nửa vòng trái đất hơn bảy năm....
  worker 3: Nửa vòng trái đất hơn bảy năm....
OK — all workers produced identical, correct transcripts
```

**4 concurrent batch calls on one shared Pipeline** ([thread_stress.py](../bindings/python/examples/thread_stress.py)):
```
Pattern A — one Pipeline per worker (parallel-safe)
  unique transcripts: 1   all identical

Pattern B — one Pipeline, no app lock (internal drive_mutex serialises)
  unique transcripts: 1   no crash, all identical
```

Pre-fix: concurrent transcribe on shared Pipeline crashed with `double free or corruption`. Post-fix: no crash, all threads produce byte-identical Vietnamese transcripts.

## Roadmap

- **Lift the batch `drive_mutex`** by making `TranscribeFile/Buffer` internally drive a temporary Session (so multiple batch calls become multiple Sessions, each with its own state). Would let concurrent batch calls run in parallel too.

## Verifying

The unit tests (`ctest`, 69 cases) are single-threaded. Concurrency is covered by the two stress scripts above:

- [`multi_session_stress.py`](../bindings/python/examples/multi_session_stress.py) — N streaming Sessions on one Pipeline must produce identical transcripts.
- [`thread_stress.py`](../bindings/python/examples/thread_stress.py) — concurrent batch calls, one Pipeline per worker and one shared Pipeline.
