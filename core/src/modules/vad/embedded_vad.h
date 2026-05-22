#ifndef VIETASR_MODULES_VAD_EMBEDDED_VAD_H
#define VIETASR_MODULES_VAD_EMBEDDED_VAD_H

#include "embedded_model.h"  // EmbeddedBlob

namespace vietasr {
namespace vad {

// ten-vad.onnx baked into libvietasr at build time. data is nullptr / size 0
// when the model was not embedded (VIETASR_EMBED_TENVAD=OFF).
EmbeddedBlob EmbeddedTenVadModel();

// silero_vad.onnx baked into libvietasr at build time. data is nullptr /
// size 0 when the model was not embedded (VIETASR_EMBED_SILERO=OFF).
EmbeddedBlob EmbeddedSileroModel();

}  // namespace vad
}  // namespace vietasr

#endif  // VIETASR_MODULES_VAD_EMBEDDED_VAD_H
