#ifndef VIETASR_MODULES_ITN_ITN_NORMALIZER_H
#define VIETASR_MODULES_ITN_ITN_NORMALIZER_H

#include <string>

// Lightweight inverse-text-normalization facade. Unlike processor.h this header
// pulls in no OpenFst, so any module (e.g. vietasr, to normalize each sentence
// as it is committed during streaming) can include it cheaply.

namespace vietasr {
namespace itn {

// Inverse-normalize spoken-form Vietnamese into written form
// ("hai mươi nghìn đồng" -> "20000₫"). The embedded FST grammar is loaded once,
// lazily, on first use and shared process-wide (thread-safe afterwards — the
// grammar is read-only). Returns `spoken` unchanged when ITN is unavailable.
std::string NormalizeText(const std::string& spoken);

// True once the embedded grammar has loaded successfully.
bool ItnAvailable();

}  // namespace itn
}  // namespace vietasr

#endif  // VIETASR_MODULES_ITN_ITN_NORMALIZER_H
