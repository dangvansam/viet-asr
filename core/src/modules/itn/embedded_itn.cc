#include "modules/itn/embedded_itn.h"

#if VIETASR_HAS_EMBEDDED_ITN

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

vietasr::itn::ItnBlob LoadResource(int id) {
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
namespace itn {

// Resource ids 200 / 201 — see cmake/embedded_itn.rc.in.
ItnBlob EmbeddedItnTagger() { return LoadResource(200); }
ItnBlob EmbeddedItnVerbalizer() { return LoadResource(201); }

}  // namespace itn
}  // namespace vietasr

#else  // non-MSVC: symbols come from the .incbin assembly file

extern "C" {
extern const unsigned char vietasr_itn_tagger_fst[];
extern const unsigned char vietasr_itn_verbalizer_fst[];
extern const unsigned long long vietasr_itn_tagger_fst_size;
extern const unsigned long long vietasr_itn_verbalizer_fst_size;
}

namespace vietasr {
namespace itn {

ItnBlob EmbeddedItnTagger() {
    return {vietasr_itn_tagger_fst,
            static_cast<std::size_t>(vietasr_itn_tagger_fst_size)};
}

ItnBlob EmbeddedItnVerbalizer() {
    return {vietasr_itn_verbalizer_fst,
            static_cast<std::size_t>(vietasr_itn_verbalizer_fst_size)};
}

}  // namespace itn
}  // namespace vietasr

#endif  // _MSC_VER

#else  // VIETASR_HAS_EMBEDDED_ITN not set — grammar was not embedded

namespace vietasr {
namespace itn {

ItnBlob EmbeddedItnTagger() { return {nullptr, 0}; }
ItnBlob EmbeddedItnVerbalizer() { return {nullptr, 0}; }

}  // namespace itn
}  // namespace vietasr

#endif  // VIETASR_HAS_EMBEDDED_ITN
