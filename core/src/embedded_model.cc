#include "embedded_model.h"

#if VIETASR_HAS_EMBEDDED_MODEL

#if defined(_MSC_VER)

#include <windows.h>

#include <vector>

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

vietasr::EmbeddedBlob LoadOneResource(HMODULE module, int id) {
    if (!module) return {nullptr, 0};
    // RT_RCDATA expands to the ANSI MAKEINTRESOURCE form (LPSTR); FindResourceW
    // needs an LPCWSTR. The value is an integer-in-pointer, so reinterpret it.
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

// Model chunks are stored as resources 100, 101, ... — concatenate them once.
const std::vector<unsigned char>& ModelBytes() {
    static const std::vector<unsigned char> bytes = [] {
        std::vector<unsigned char> out;
        HMODULE module = SelfModule();
        for (int id = 100;; ++id) {
            vietasr::EmbeddedBlob chunk = LoadOneResource(module, id);
            if (chunk.empty()) break;
            out.insert(out.end(), chunk.data, chunk.data + chunk.size);
        }
        return out;
    }();
    return bytes;
}

}

namespace vietasr {

EmbeddedBlob EmbeddedAsrModel() {
    const std::vector<unsigned char>& bytes = ModelBytes();
    if (bytes.empty()) return {nullptr, 0};
    return {bytes.data(), bytes.size()};
}

EmbeddedBlob EmbeddedAsrVocab() {
    return LoadOneResource(SelfModule(), 2);
}

}

#else  // non-MSVC: symbols come from the .incbin assembly file

extern "C" {
extern const unsigned char vietasr_asr_model_onnx[];
extern const unsigned char vietasr_asr_vocab[];
extern const unsigned long long vietasr_asr_model_onnx_size;
extern const unsigned long long vietasr_asr_vocab_size;
}

namespace vietasr {

EmbeddedBlob EmbeddedAsrModel() {
    return {vietasr_asr_model_onnx,
            static_cast<std::size_t>(vietasr_asr_model_onnx_size)};
}

EmbeddedBlob EmbeddedAsrVocab() {
    return {vietasr_asr_vocab,
            static_cast<std::size_t>(vietasr_asr_vocab_size)};
}

}

#endif  // _MSC_VER

#else  // VIETASR_HAS_EMBEDDED_MODEL not set — model was not embedded

namespace vietasr {

EmbeddedBlob EmbeddedAsrModel() { return {nullptr, 0}; }
EmbeddedBlob EmbeddedAsrVocab() { return {nullptr, 0}; }

}

#endif
