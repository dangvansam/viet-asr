#include "modules/itn/itn_normalizer.h"

#include <mutex>

#include "modules/itn/embedded_itn.h"
#include "modules/itn/processor.h"

namespace vietasr {
namespace itn {

namespace {

// Process-wide ITN processor, loaded once from the embedded grammar. The FSTs
// are read-only after load, so concurrent NormalizeText() calls are safe.
const ItnProcessor& SharedProcessor() {
    static ItnProcessor processor;
    static std::once_flag once;
    std::call_once(once, [] {
        const ItnBlob tagger = EmbeddedItnTagger();
        const ItnBlob verbalizer = EmbeddedItnVerbalizer();
        processor.Load(tagger.data, tagger.size,
                       verbalizer.data, verbalizer.size);
    });
    return processor;
}

}  // namespace

std::string NormalizeText(const std::string& spoken) {
    const ItnProcessor& processor = SharedProcessor();
    return processor.ready() ? processor.Normalize(spoken) : spoken;
}

bool ItnAvailable() {
    return SharedProcessor().ready();
}

}  // namespace itn
}  // namespace vietasr
