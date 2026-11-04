#include "modules/vietasr/transducer_decoder.h"

#include <utility>

namespace vietasr {

TransducerDecoder::TransducerDecoder(std::shared_ptr<const TransducerModel> model,
                                     const TransducerSearchConfig& config)
    : model_(std::move(model)), config_(config) {}

Status TransducerDecoder::Decode(const EncoderOutput& encoder_out,
                                 std::vector<int>* tokens,
                                 std::vector<int>* frames) const {
    tokens->clear();
    if (!model_) return Status::Error(-6, "transducer: model not loaded");
    TransducerBeamSearch search(config_);
    search.Begin();
    std::vector<float> decoder_out;
    std::vector<float> encoder_rows;
    std::vector<float> decoder_rows;
    std::vector<float> logits;
    for (int t = 0; t < encoder_out.frames; ++t) {
        std::vector<std::int64_t> pending = search.PendingContexts();
        if (!pending.empty()) {
            int batch = static_cast<int>(pending.size()) / model_->context_size();
            Status status = model_->Decode(pending, batch, &decoder_out);
            if (!status.ok()) return status;
            search.AcceptDecoderOutputs(pending, decoder_out);
        }
        int batch = search.BuildJoinerInputs(encoder_out.Frame(t), encoder_out.dim,
                                             &encoder_rows, &decoder_rows);
        if (batch == 0) return Status::Error(-6, "transducer: missing decoder state");
        Status status = model_->Join(encoder_rows, decoder_rows, batch, &logits);
        if (!status.ok()) return status;
        search.AcceptLogits(logits, batch);
    }
    *tokens = search.BestTokens();
    if (frames != nullptr) *frames = search.BestFrames();
    return Status::Ok();
}

}
