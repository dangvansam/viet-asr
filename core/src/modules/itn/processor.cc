// Adapted from FunASR (runtime/onnxruntime/src/itn-processor.cpp) /
// WeTextProcessing. See processor.h for the original copyright + Apache-2.0
// notice. glog/exit() error handling has been replaced with VIETASR_LOG_* and
// graceful return values; the WeText token-reorder step is dropped (see the
// note in processor.h).

#include "modules/itn/processor.h"

#include <ios>
#include <sstream>

#include "vietasr/logger.h"

namespace vietasr {
namespace itn {

namespace {

// Reads an OpenFst archive from a byte buffer into a StdVectorFst.
std::shared_ptr<fst::StdVectorFst> ReadFst(const unsigned char* data,
                                           std::size_t size,
                                           const char* source) {
    if (data == nullptr || size == 0) return nullptr;
    const std::string buffer(reinterpret_cast<const char*>(data), size);
    std::istringstream strm(buffer,
                            std::ios_base::in | std::ios_base::binary);
    std::unique_ptr<fst::StdVectorFst> fst(
        fst::StdVectorFst::Read(strm, fst::FstReadOptions(source)));
    if (fst) {
        // Compose() matches input_fst's output labels against the grammar's
        // input labels, so the grammar must be input-label sorted. Sorting at
        // load time keeps it correct regardless of how the FST was exported.
        fst::ArcSort(fst.get(), fst::ILabelCompare<fst::StdArc>());
    }
    return std::shared_ptr<fst::StdVectorFst>(std::move(fst));
}

}  // namespace

bool ItnProcessor::Load(const unsigned char* tagger_data,
                        std::size_t tagger_size,
                        const unsigned char* verbalizer_data,
                        std::size_t verbalizer_size) {
    tagger_ = ReadFst(tagger_data, tagger_size, "itn_tagger");
    verbalizer_ = ReadFst(verbalizer_data, verbalizer_size, "itn_verbalizer");
    if (!tagger_) {
        VIETASR_LOG_ERROR("itn") << "failed to read ITN tagger grammar";
    }
    if (!verbalizer_) {
        VIETASR_LOG_ERROR("itn") << "failed to read ITN verbalizer grammar";
    }
    return ready();
}

std::string ItnProcessor::ShortestPathString(
    const fst::StdVectorFst& lattice) {
    if (lattice.Start() == fst::kNoStateId) return "";
    fst::StdVectorFst path;
    fst::ShortestPath(lattice, &path, /*nshortest=*/1, /*unique=*/true);
    if (path.Start() == fst::kNoStateId) return "";
    std::string output;
    fst::StringPrinter<fst::StdArc> printer(fst::TokenType::BYTE);
    printer(path, &output);
    return output;
}

std::string ItnProcessor::Compose(const std::string& input,
                                  const fst::StdVectorFst& grammar) const {
    fst::StdVectorFst input_fst;
    fst::StringCompiler<fst::StdArc> compiler(fst::TokenType::BYTE);
    if (!compiler(input, &input_fst)) return "";
    fst::StdVectorFst lattice;
    fst::Compose(input_fst, grammar, &lattice);
    return ShortestPathString(lattice);
}

std::string ItnProcessor::Tag(const std::string& input) const {
    return Compose(input, *tagger_);
}

std::string ItnProcessor::Verbalize(const std::string& tagged) const {
    if (tagged.empty()) return "";
    return Compose(tagged, *verbalizer_);
}

std::string ItnProcessor::Normalize(const std::string& input) const {
    if (!ready() || input.empty()) return input;
    const std::string result = Verbalize(Tag(input));
    return result.empty() ? input : result;
}

}  // namespace itn
}  // namespace vietasr
