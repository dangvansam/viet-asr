#ifndef VIETASR_EMBEDDED_MODEL_H
#define VIETASR_EMBEDDED_MODEL_H

#include <cstddef>

namespace vietasr {

// A read-only view of a blob baked into libvietasr at build time.
// data is nullptr / size is 0 when the model was not embedded
// (i.e. VIETASR_EMBED_MODEL=OFF).
struct EmbeddedBlob final {
    const unsigned char* data;
    std::size_t size;

    bool empty() const { return data == nullptr || size == 0; }
};

// The chunked vietasr ONNX model, reassembled and embedded by EmbedModel.cmake.
EmbeddedBlob EmbeddedAsrModel();

// The vietasr BPE vocab (vocab.txt), embedded alongside the model.
EmbeddedBlob EmbeddedAsrVocab();

}

#endif
