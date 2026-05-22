#ifndef VIETASR_MODULES_ITN_ITN_VI_H
#define VIETASR_MODULES_ITN_ITN_VI_H

#include <string>

namespace vietasr {

// Rule-based inverse text normalization for Vietnamese spoken numbers.
//
// Folds spoken numerals into digits, e.g.
//   "bốn mươi"               -> "40"
//   "một trăm hai mươi lăm"  -> "125"
//   "bốn mươi phần trăm"     -> "40%"
//   "ba phẩy năm"            -> "3.5"
//
// Pure: depends only on the C++ standard library — no filesystem, no threads,
// no model. Safe to compile into WebAssembly. Idempotent (running it again on
// already-normalized text is a no-op) and safe on partial/streaming transcripts
// (a numeral still being spoken folds to an intermediate value and self-corrects
// once the numeral completes).
std::string ApplyItnVietnamese(const std::string& text);

}  // namespace vietasr

#endif  // VIETASR_MODULES_ITN_ITN_VI_H
