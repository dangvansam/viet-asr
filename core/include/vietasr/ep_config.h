#ifndef VIETASR_EP_CONFIG_H
#define VIETASR_EP_CONFIG_H

namespace vietasr {

// Execution-provider request carried from the C ABI down to the ONNX engine.
// ep_enum holds a VietasrExecutionProvider value (0 == VIETASR_EP_AUTO).
struct EpPreference final {
    int ep_enum{0};
    int device_id{0};
};

// Short, lower-case name for a VietasrExecutionProvider enum value
// (e.g. "coreml", "cuda"). Returns "unknown" for out-of-range values.
const char* EpEnumName(int ep_enum);

}

#endif
