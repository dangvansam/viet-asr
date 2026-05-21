#ifndef VIETASR_MODULES_ITN_EMBEDDED_ITN_H
#define VIETASR_MODULES_ITN_EMBEDDED_ITN_H

#include <cstddef>

namespace vietasr {
namespace itn {

// A read-only view of an ITN grammar baked into libvietasr at build time.
// data is nullptr / size is 0 when the grammar was not embedded
// (i.e. VIETASR_EMBED_ITN=OFF).
struct ItnBlob final {
    const unsigned char* data;
    std::size_t size;

    bool empty() const { return data == nullptr || size == 0; }
};

// The Vietnamese ITN tagger grammar (vi_itn_tagger.fst).
ItnBlob EmbeddedItnTagger();

// The Vietnamese ITN verbalizer grammar (vi_itn_verbalizer.fst).
ItnBlob EmbeddedItnVerbalizer();

}  // namespace itn
}  // namespace vietasr

#endif  // VIETASR_MODULES_ITN_EMBEDDED_ITN_H
