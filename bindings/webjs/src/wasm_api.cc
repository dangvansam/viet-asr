#include <emscripten/emscripten.h>

#include <memory>
#include <string>
#include <vector>

#include "modules/vietasr/ctc_beam_search.h"
#include "modules/vietasr/post_processor.h"
#include "modules/vietasr/units.h"
#include "preprocess/audio_resampler.h"
#include "preprocess/feature_pipeline.h"

using vietasr::CtcBeamSearch;
using vietasr::CtcBeamSearchConfig;
using vietasr::FeaturePipeline;
using vietasr::PostProcessor;
using vietasr::Units;

namespace {

constexpr int kBlankId = 0;
constexpr int kFeatureDim = 80;

struct WasmState {
    std::unique_ptr<FeaturePipeline> features;
    std::unique_ptr<CtcBeamSearch> beam;
    Units units;
    PostProcessor post;
    std::string result;
    std::vector<float> feature_scratch;
};

WasmState& State() {
    static WasmState state;
    return state;
}

}

extern "C" {

EMSCRIPTEN_KEEPALIVE
int vietasr_wasm_init(const char* units_text) {
    WasmState& s = State();
    s.features = std::make_unique<FeaturePipeline>();
    s.beam = std::make_unique<CtcBeamSearch>(CtcBeamSearchConfig{10, kBlankId});
    auto status = s.units.LoadFromText(units_text ? units_text : "");
    return status.ok() ? 0 : status.code();
}

EMSCRIPTEN_KEEPALIVE
int vietasr_wasm_vocab_size() {
    return State().units.size();
}

EMSCRIPTEN_KEEPALIVE
void vietasr_wasm_reset() {
    WasmState& s = State();
    if (s.features) s.features->Reset();
    if (s.beam) s.beam->Reset();
    s.feature_scratch.clear();
}

/* Push raw PCM (float32, any sample rate). Resamples to 16 kHz and runs
 * the fbank front-end. Returns the number of feature frames now ready. */
EMSCRIPTEN_KEEPALIVE
int vietasr_wasm_accept_pcm(const float* pcm, int len, int sample_rate) {
    WasmState& s = State();
    if (!s.features) return -1;
    if (sample_rate != 16000) {
        vietasr::AudioResampler resampler(sample_rate, 16000);
        resampler.AcceptWaveform(pcm, static_cast<std::size_t>(len));
        resampler.Flush();
        std::vector<float> resampled = resampler.PopAll();
        s.features->AcceptWaveform(resampled.data(), resampled.size());
    } else {
        s.features->AcceptWaveform(pcm, static_cast<std::size_t>(len));
    }
    return static_cast<int>(s.features->num_frames_ready());
}

EMSCRIPTEN_KEEPALIVE
int vietasr_wasm_frames_ready() {
    WasmState& s = State();
    return s.features ? static_cast<int>(s.features->num_frames_ready()) : 0;
}

/* Pop up to `n` feature frames into an internal buffer; returns a pointer
 * to [n_frames * 80] floats. Caller reads via HEAPF32. */
EMSCRIPTEN_KEEPALIVE
const float* vietasr_wasm_pop_features(int n) {
    WasmState& s = State();
    s.feature_scratch = s.features->PopFrames(static_cast<std::size_t>(n));
    return s.feature_scratch.data();
}

EMSCRIPTEN_KEEPALIVE
int vietasr_wasm_feature_dim() {
    return kFeatureDim;
}

/* Feed one chunk of CTC logits ([n_frames * vocab_size] floats) into the
 * beam search. JS obtains these from onnxruntime-web. */
EMSCRIPTEN_KEEPALIVE
void vietasr_wasm_decode_logits(const float* logits, int n_frames, int vocab_size) {
    WasmState& s = State();
    if (!s.beam) return;
    for (int f = 0; f < n_frames; ++f) {
        s.beam->Step(logits + static_cast<std::size_t>(f) * vocab_size, vocab_size);
    }
}

EMSCRIPTEN_KEEPALIVE
const char* vietasr_wasm_transcript() {
    WasmState& s = State();
    if (!s.beam) {
        s.result.clear();
        return s.result.c_str();
    }
    std::vector<int> ids = s.beam->Hypothesis();
    std::vector<std::string> tokens;
    tokens.reserve(ids.size());
    for (int id : ids) {
        if (id == kBlankId) continue;
        tokens.push_back(s.units.At(id));
    }
    s.result = s.post.Detokenize(tokens);
    return s.result.c_str();
}

}
