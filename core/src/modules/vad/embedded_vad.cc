#include "modules/vad/embedded_vad.h"

#include <stddef.h>

#if defined(_MSC_VER)

#include <windows.h>

namespace {

HMODULE SelfModule() {
    HMODULE module = nullptr;
    GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
            | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&SelfModule),
        &module);
    return module;
}

vietasr::EmbeddedBlob LoadResource(int id) {
    HMODULE module = SelfModule();
    if (!module) return {nullptr, 0};
    HRSRC info = FindResourceW(module, MAKEINTRESOURCEW(id),
                               reinterpret_cast<LPCWSTR>(RT_RCDATA));
    if (!info) return {nullptr, 0};
    HGLOBAL handle = LoadResource(module, info);
    if (!handle) return {nullptr, 0};
    const void* data = LockResource(handle);
    DWORD size = SizeofResource(module, info);
    if (!data || size == 0) return {nullptr, 0};
    return {static_cast<const unsigned char*>(data),
            static_cast<std::size_t>(size)};
}

}  // namespace

namespace vietasr {
namespace vad {

// Resource ids 300 / 301 — see cmake/embedded_vad.rc.in.
EmbeddedBlob EmbeddedTenVadModel() {
#if defined(VIETASR_HAS_EMBEDDED_TENVAD)
    return LoadResource(300);
#else
    return {nullptr, 0};
#endif
}

EmbeddedBlob EmbeddedSileroModel() {
#if defined(VIETASR_HAS_EMBEDDED_SILERO)
    return LoadResource(301);
#else
    return {nullptr, 0};
#endif
}

}  // namespace vad
}  // namespace vietasr

#else  // non-MSVC: symbols come from the .incbin assembly file

#if defined(VIETASR_HAS_EMBEDDED_TENVAD)
extern "C" {
extern const unsigned char vietasr_tenvad_onnx[];
extern const unsigned long long vietasr_tenvad_onnx_size;
}
#endif

#if defined(VIETASR_HAS_EMBEDDED_SILERO)
extern "C" {
extern const unsigned char vietasr_silero_onnx[];
extern const unsigned long long vietasr_silero_onnx_size;
}
#endif

namespace vietasr {
namespace vad {

EmbeddedBlob EmbeddedTenVadModel() {
#if defined(VIETASR_HAS_EMBEDDED_TENVAD)
    return {vietasr_tenvad_onnx,
            static_cast<std::size_t>(vietasr_tenvad_onnx_size)};
#else
    return {nullptr, 0};
#endif
}

EmbeddedBlob EmbeddedSileroModel() {
#if defined(VIETASR_HAS_EMBEDDED_SILERO)
    return {vietasr_silero_onnx,
            static_cast<std::size_t>(vietasr_silero_onnx_size)};
#else
    return {nullptr, 0};
#endif
}

}  // namespace vad
}  // namespace vietasr

#endif  // _MSC_VER

// C-linkage accessors for the TEN-VAD model blob. The vendored TEN-VAD source
// (third_party/ten-vad/src/aed.cc, patched) calls these to load the model from
// memory instead of a working-directory-relative file path.
extern "C" {

const void* vietasr_tenvad_model_data(void) {
    return vietasr::vad::EmbeddedTenVadModel().data;
}

size_t vietasr_tenvad_model_size(void) {
    return vietasr::vad::EmbeddedTenVadModel().size;
}

}  // extern "C"
