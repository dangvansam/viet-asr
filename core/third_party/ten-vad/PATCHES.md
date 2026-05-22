# Vendored TEN-VAD — vietasr patches

Upstream: https://github.com/TEN-framework/ten-vad
Pinned commit: see [COMMIT](COMMIT) — `22a3bcd4509d0faaa8eef4881e8af5f39c178950`

TEN-VAD's C/C++ sources are compiled directly into `libvietasr` (the `tenvad`
backend of the `vad` module). TEN-VAD already uses the ONNX Runtime **C API**,
the same runtime vietasr bundles, so no extra dependency is introduced.

## What is vendored

- `include/ten_vad.h` — the public C API.
- `src/*.cc`, `src/fftw.c`, `src/*.h`, `src/*_st.h`, `src/coeff.h` — the
  implementation, compiled into `libvietasr`.
- `src/onnx_model/ten-vad.onnx` — the model, embedded into `libvietasr` by
  `core/cmake/EmbedVadModel.cmake`.
- `LICENSE`, `NOTICES` — kept verbatim.

**Not vendored:** the upstream `lib/` prebuilt binaries, `examples*/`,
`testset/`, Python/Java/Web wrappers — vietasr builds from source.

## Patches applied

Upstream loads the model with a hardcoded relative path
(`"onnx_model/ten-vad.onnx"`), which only works when the process working
directory happens to contain that file. vietasr embeds the model into the
library instead, so two source files are patched. Every change is marked with a
`// vietasr patch:` comment.

### `src/aed_st.h`
- Declared a second `AUP_MODULE_AIVAD` constructor taking an in-memory model
  buffer: `AUP_MODULE_AIVAD(const void* model_data, size_t model_size)`.
- Declared a private `InitFromSession()` helper.

### `src/aed.cc`
- Extracted the post-session initialization (input/output discovery, tensor
  allocation) from the original constructor into `InitFromSession()`.
- The original `AUP_MODULE_AIVAD(char* onnx_path)` constructor is unchanged in
  behavior — it still calls `CreateSession` with a file path, then delegates to
  `InitFromSession()`.
- Added the buffer constructor, identical except it calls
  `CreateSessionFromArray` instead of `CreateSession`.
- At the `AUP_MODULE_AIVAD` creation site, guarded with
  `#if defined(VIETASR_HAS_EMBEDDED_TENVAD)`: when the model is embedded, the
  buffer constructor is used with `vietasr_tenvad_model_data()` /
  `vietasr_tenvad_model_size()` (defined in
  `core/src/modules/vad/embedded_vad.cc`). The `#else` branch keeps the
  original file-path behavior unchanged.

## License note

TEN-VAD is licensed under **Apache-2.0 with additional conditions** — notably a
clause prohibiting deployment "in a way that competes with Agora's offerings",
and a requested "Powered by ten-vad" attribution. See [LICENSE](LICENSE) and
[NOTICES](NOTICES). This is **not** plain Apache-2.0; review the terms before
redistributing. The attribution is carried in
`core/src/modules/vad/README.md` and the project README
(Third-party notices section).
