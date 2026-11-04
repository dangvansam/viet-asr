#ifndef VIETASR_EMBEDDED_MODEL_H
#define VIETASR_EMBEDDED_MODEL_H

#include <cstddef>

namespace vietasr {

struct EmbeddedBlob final {
    const unsigned char* data;
    std::size_t size;

    bool empty() const { return data == nullptr || size == 0; }
};

EmbeddedBlob EmbeddedAsrEncoder();
EmbeddedBlob EmbeddedAsrDecoder();
EmbeddedBlob EmbeddedAsrJoiner();
EmbeddedBlob EmbeddedAsrTokens();

}

#endif
