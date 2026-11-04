#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "modules/vietasr/post_processor.h"
#include "modules/vietasr/transducer_search.h"
#include "modules/vietasr/units.h"
#include "preprocess/audio_resampler.h"
#include "preprocess/kaldi_fbank.h"

namespace {

constexpr int kSampleRate = 16000;
constexpr int kContextSize = 2;
constexpr int kBlankId = 0;
constexpr int kUnkId = 2;

class WasmRecognizerState final {
public:
    vietasr::KaldiFbank fbank;
    vietasr::Units units;
    vietasr::PostProcessor post;
    std::unique_ptr<vietasr::TransducerBeamSearch> search;
    int max_active_paths{4};

    std::vector<float> pcm;
    std::vector<float> features;
    std::vector<std::int64_t> pending;
    std::vector<std::int32_t> pending_i32;
    std::vector<float> joiner_encoder;
    std::vector<float> joiner_decoder;
    std::string text;

    static WasmRecognizerState& Instance() {
        static WasmRecognizerState state;
        return state;
    }
};

}

extern "C" {

EMSCRIPTEN_KEEPALIVE
int vietasr_wasm_init(const char* tokens_text, int max_active_paths) {
    WasmRecognizerState& s = WasmRecognizerState::Instance();
    s.max_active_paths = max_active_paths > 0 ? max_active_paths : 4;
    auto status = s.units.LoadFromText(tokens_text ? tokens_text : "");
    return status.ok() ? 0 : status.code();
}

EMSCRIPTEN_KEEPALIVE
int vietasr_wasm_vocab_size() {
    return WasmRecognizerState::Instance().units.size();
}

EMSCRIPTEN_KEEPALIVE
int vietasr_wasm_feature_dim() {
    return WasmRecognizerState::Instance().fbank.num_bins();
}

EMSCRIPTEN_KEEPALIVE
void vietasr_wasm_reset() {
    WasmRecognizerState& s = WasmRecognizerState::Instance();
    s.pcm.clear();
    s.features.clear();
    s.search.reset();
}

EMSCRIPTEN_KEEPALIVE
int vietasr_wasm_accept_pcm(const float* pcm, int len, int sample_rate) {
    WasmRecognizerState& s = WasmRecognizerState::Instance();
    if (pcm == nullptr || len <= 0) return static_cast<int>(s.pcm.size());
    if (sample_rate != kSampleRate) {
        vietasr::AudioResampler resampler(sample_rate, kSampleRate);
        resampler.AcceptWaveform(pcm, static_cast<std::size_t>(len));
        resampler.Flush();
        std::vector<float> resampled = resampler.PopAll();
        s.pcm.insert(s.pcm.end(), resampled.begin(), resampled.end());
    } else {
        s.pcm.insert(s.pcm.end(), pcm, pcm + len);
    }
    return static_cast<int>(s.pcm.size());
}

EMSCRIPTEN_KEEPALIVE
int vietasr_wasm_pcm_length() {
    return static_cast<int>(WasmRecognizerState::Instance().pcm.size());
}

EMSCRIPTEN_KEEPALIVE
const float* vietasr_wasm_pcm_ptr() {
    return WasmRecognizerState::Instance().pcm.data();
}

EMSCRIPTEN_KEEPALIVE
int vietasr_wasm_compute_features(int start, int len) {
    WasmRecognizerState& s = WasmRecognizerState::Instance();
    int total = static_cast<int>(s.pcm.size());
    if (start < 0) start = 0;
    if (start > total) start = total;
    if (len < 0 || start + len > total) len = total - start;
    s.features = s.fbank.Compute(s.pcm.data() + start, static_cast<std::size_t>(len));
    return static_cast<int>(s.features.size() / static_cast<std::size_t>(s.fbank.num_bins()));
}

EMSCRIPTEN_KEEPALIVE
const float* vietasr_wasm_features_ptr() {
    return WasmRecognizerState::Instance().features.data();
}

EMSCRIPTEN_KEEPALIVE
void vietasr_wasm_search_begin() {
    WasmRecognizerState& s = WasmRecognizerState::Instance();
    s.search = std::make_unique<vietasr::TransducerBeamSearch>(
        vietasr::TransducerSearchConfig{s.max_active_paths, kContextSize, kBlankId, kUnkId});
    s.search->Begin();
}

EMSCRIPTEN_KEEPALIVE
int vietasr_wasm_search_pending() {
    WasmRecognizerState& s = WasmRecognizerState::Instance();
    if (!s.search) return 0;
    s.pending = s.search->PendingContexts();
    s.pending_i32.assign(s.pending.begin(), s.pending.end());
    return static_cast<int>(s.pending.size()) / kContextSize;
}

EMSCRIPTEN_KEEPALIVE
const std::int32_t* vietasr_wasm_search_pending_ptr() {
    return WasmRecognizerState::Instance().pending_i32.data();
}

EMSCRIPTEN_KEEPALIVE
void vietasr_wasm_search_accept_decoder(const float* outputs, int batch, int dim) {
    WasmRecognizerState& s = WasmRecognizerState::Instance();
    if (!s.search || outputs == nullptr || batch <= 0) return;
    std::vector<float> data(outputs, outputs + static_cast<std::size_t>(batch) * static_cast<std::size_t>(dim));
    s.search->AcceptDecoderOutputs(s.pending, data);
}

EMSCRIPTEN_KEEPALIVE
int vietasr_wasm_search_joiner_inputs(const float* encoder_frame, int dim) {
    WasmRecognizerState& s = WasmRecognizerState::Instance();
    if (!s.search) return 0;
    return s.search->BuildJoinerInputs(encoder_frame, dim, &s.joiner_encoder, &s.joiner_decoder);
}

EMSCRIPTEN_KEEPALIVE
const float* vietasr_wasm_joiner_encoder_ptr() {
    return WasmRecognizerState::Instance().joiner_encoder.data();
}

EMSCRIPTEN_KEEPALIVE
const float* vietasr_wasm_joiner_decoder_ptr() {
    return WasmRecognizerState::Instance().joiner_decoder.data();
}

EMSCRIPTEN_KEEPALIVE
void vietasr_wasm_search_accept_logits(const float* logits, int batch, int vocab) {
    WasmRecognizerState& s = WasmRecognizerState::Instance();
    if (!s.search || logits == nullptr || batch <= 0) return;
    std::vector<float> data(logits, logits + static_cast<std::size_t>(batch) * static_cast<std::size_t>(vocab));
    s.search->AcceptLogits(data, batch);
}

EMSCRIPTEN_KEEPALIVE
const char* vietasr_wasm_search_text() {
    WasmRecognizerState& s = WasmRecognizerState::Instance();
    s.text.clear();
    if (!s.search) return s.text.c_str();
    std::vector<std::string> tokens;
    for (int id : s.search->BestTokens()) tokens.push_back(s.units.At(id));
    s.text = s.post.Detokenize(tokens);
    return s.text.c_str();
}

}
