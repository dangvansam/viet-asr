// WFST inverse text normalizer: composes input against a tagger grammar, then
// a verbalizer grammar, to turn spoken-form text into written form, e.g.
//   "hai mươi nghìn đồng" -> "20000₫"
//
// Adapted from FunASR (runtime/onnxruntime/src/itn-processor.h), itself adapted
// from WeTextProcessing. Original work:
//   Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
//                 2023 Jing Du (thuduj12@163.com)
//   Licensed under the Apache License, Version 2.0.
//
// NOTE: FunASR's runtime processor reorders the tagged token stream before
// verbalizing (WeTextProcessing's zh grammars need it). The fun_text_processing
// Vietnamese grammar is NeMo-derived: its tagger already emits members in the
// order the verbalizer expects, so the tagged text is verbalized directly.

#ifndef VIETASR_MODULES_ITN_PROCESSOR_H
#define VIETASR_MODULES_ITN_PROCESSOR_H

#include <cstddef>
#include <memory>
#include <string>

#include "fst/fstlib.h"

namespace vietasr {
namespace itn {

class ItnProcessor {
public:
    ItnProcessor() = default;

    // Loads the tagger + verbalizer FSTs from in-memory OpenFst archives
    // (the byte layout produced by VectorFst::Write). Returns false if either
    // archive fails to parse.
    bool Load(const unsigned char* tagger_data, std::size_t tagger_size,
              const unsigned char* verbalizer_data,
              std::size_t verbalizer_size);

    bool ready() const { return tagger_ != nullptr && verbalizer_ != nullptr; }

    // Inverse-normalizes spoken-form text into written form. Returns `input`
    // unchanged if the processor is not loaded or the grammar produces nothing.
    // Thread-safe once Load() has succeeded (the FSTs are read-only).
    std::string Normalize(const std::string& input) const;

private:
    std::string Tag(const std::string& input) const;
    std::string Verbalize(const std::string& tagged) const;
    std::string Compose(const std::string& input,
                        const fst::StdVectorFst& grammar) const;
    static std::string ShortestPathString(const fst::StdVectorFst& lattice);

    std::shared_ptr<fst::StdVectorFst> tagger_;
    std::shared_ptr<fst::StdVectorFst> verbalizer_;
};

}  // namespace itn
}  // namespace vietasr

#endif  // VIETASR_MODULES_ITN_PROCESSOR_H
