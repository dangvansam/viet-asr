#include "embedded_model.h"

#if VIETASR_HAS_EMBEDDED_MODEL

#if defined(_MSC_VER)

#include <windows.h>

#include <vector>

namespace {

HMODULE SelfModule() {
    HMODULE module = nullptr;
    GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&SelfModule),
        &module);
    return module;
}

vietasr::EmbeddedBlob LoadOneResource(HMODULE module, int id) {
    if (!module) return {nullptr, 0};
    HRSRC info = FindResourceW(module, MAKEINTRESOURCEW(id), reinterpret_cast<LPCWSTR>(RT_RCDATA));
    if (!info) return {nullptr, 0};
    HGLOBAL handle = LoadResource(module, info);
    if (!handle) return {nullptr, 0};
    const void* data = LockResource(handle);
    DWORD size = SizeofResource(module, info);
    if (!data || size == 0) return {nullptr, 0};
    return {static_cast<const unsigned char*>(data), static_cast<std::size_t>(size)};
}

class ResourceAssembler final {
public:
    explicit ResourceAssembler(int base) {
        HMODULE module = SelfModule();
        for (int id = base; id < base + 100; ++id) {
            vietasr::EmbeddedBlob chunk = LoadOneResource(module, id);
            if (chunk.empty()) break;
            bytes.insert(bytes.end(), chunk.data, chunk.data + chunk.size);
        }
    }

    vietasr::EmbeddedBlob Blob() const {
        if (bytes.empty()) return {nullptr, 0};
        return {bytes.data(), bytes.size()};
    }

    std::vector<unsigned char> bytes;
};

}

namespace vietasr {

EmbeddedBlob EmbeddedAsrEncoder() {
    static const ResourceAssembler assembled(1000);
    return assembled.Blob();
}

EmbeddedBlob EmbeddedAsrDecoder() {
    static const ResourceAssembler assembled(1100);
    return assembled.Blob();
}

EmbeddedBlob EmbeddedAsrJoiner() {
    static const ResourceAssembler assembled(1200);
    return assembled.Blob();
}

EmbeddedBlob EmbeddedAsrTokens() {
    return LoadOneResource(SelfModule(), 2);
}

}

#else

extern "C" {
extern const unsigned char vietasr_asr_encoder[];
extern const unsigned char vietasr_asr_decoder[];
extern const unsigned char vietasr_asr_joiner[];
extern const unsigned char vietasr_asr_tokens[];
extern const unsigned long long vietasr_asr_encoder_size;
extern const unsigned long long vietasr_asr_decoder_size;
extern const unsigned long long vietasr_asr_joiner_size;
extern const unsigned long long vietasr_asr_tokens_size;
}

namespace vietasr {

EmbeddedBlob EmbeddedAsrEncoder() {
    return {vietasr_asr_encoder, static_cast<std::size_t>(vietasr_asr_encoder_size)};
}

EmbeddedBlob EmbeddedAsrDecoder() {
    return {vietasr_asr_decoder, static_cast<std::size_t>(vietasr_asr_decoder_size)};
}

EmbeddedBlob EmbeddedAsrJoiner() {
    return {vietasr_asr_joiner, static_cast<std::size_t>(vietasr_asr_joiner_size)};
}

EmbeddedBlob EmbeddedAsrTokens() {
    return {vietasr_asr_tokens, static_cast<std::size_t>(vietasr_asr_tokens_size)};
}

}

#endif

#else

namespace vietasr {

EmbeddedBlob EmbeddedAsrEncoder() { return {nullptr, 0}; }
EmbeddedBlob EmbeddedAsrDecoder() { return {nullptr, 0}; }
EmbeddedBlob EmbeddedAsrJoiner() { return {nullptr, 0}; }
EmbeddedBlob EmbeddedAsrTokens() { return {nullptr, 0}; }

}

#endif
