#include "modules/vietasr/transducer_model.h"

#include <cstdlib>
#include <fstream>
#include <iterator>

#include "embedded_model.h"
#include <vietasr.h>
#include "vietasr/logger.h"

namespace vietasr {

bool TransducerAssetLoader::ReadFile(const std::string& path,
                                     std::vector<unsigned char>* out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    out->assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return !out->empty();
}

std::string TransducerAssetLoader::FirstExisting(const std::string& dir,
                                                 const std::vector<std::string>& names) {
    for (const auto& name : names) {
        std::string path = dir + "/" + name;
        std::ifstream probe(path, std::ios::binary);
        if (probe.good()) return path;
    }
    return std::string();
}

TransducerAssets TransducerAssetLoader::FromEmbedded() {
    TransducerAssets assets;
    EmbeddedBlob encoder = EmbeddedAsrEncoder();
    EmbeddedBlob decoder = EmbeddedAsrDecoder();
    EmbeddedBlob joiner = EmbeddedAsrJoiner();
    EmbeddedBlob tokens = EmbeddedAsrTokens();
    assets.encoder.data = encoder.data;
    assets.encoder.size = encoder.size;
    assets.decoder.data = decoder.data;
    assets.decoder.size = decoder.size;
    assets.joiner.data = joiner.data;
    assets.joiner.size = joiner.size;
    if (!tokens.empty()) {
        assets.tokens.assign(reinterpret_cast<const char*>(tokens.data), tokens.size);
    }
    assets.origin = "embedded";
    return assets;
}

TransducerAssets TransducerAssetLoader::FromDirectory(const std::string& dir) {
    TransducerAssets assets;
    if (dir.empty()) return assets;
    std::string encoder = FirstExisting(dir, {"encoder.int8.onnx", "encoder.onnx"});
    std::string decoder = FirstExisting(dir, {"decoder.int8.onnx", "decoder.onnx"});
    std::string joiner = FirstExisting(dir, {"joiner.int8.onnx", "joiner.onnx"});
    std::string tokens = FirstExisting(dir, {"tokens.txt"});
    if (encoder.empty() || decoder.empty() || joiner.empty() || tokens.empty()) {
        return assets;
    }
    std::vector<unsigned char> token_bytes;
    if (!ReadFile(encoder, &assets.encoder.owned)
            || !ReadFile(decoder, &assets.decoder.owned)
            || !ReadFile(joiner, &assets.joiner.owned)
            || !ReadFile(tokens, &token_bytes)) {
        return TransducerAssets{};
    }
    assets.tokens.assign(token_bytes.begin(), token_bytes.end());
    assets.origin = dir;
    return assets;
}

TransducerAssets TransducerAssetLoader::Resolve(const std::string& configured_dir,
                                                const std::string& cache_dir) {
    if (!configured_dir.empty()) {
        TransducerAssets assets = FromDirectory(configured_dir);
        if (assets.complete()) return assets;
        VIETASR_LOG_WARN("vietasr") << "model_dir has no complete model: " << configured_dir;
    }
    const char* env = std::getenv("VIETASR_ASR_MODEL_DIR");
    if (env != nullptr && *env != '\0') {
        TransducerAssets assets = FromDirectory(env);
        if (assets.complete()) return assets;
    }
    TransducerAssets embedded = FromEmbedded();
    if (embedded.complete()) return embedded;
    if (!cache_dir.empty()) {
        for (const char* sub : {"/vietasr/2", "/vietasr"}) {
            TransducerAssets cached = FromDirectory(cache_dir + sub);
            if (cached.complete()) return cached;
        }
    }
    return TransducerAssets{};
}

TransducerModel::TransducerModel() = default;
TransducerModel::~TransducerModel() = default;

Status TransducerModel::Load(Engine* encoder_engine, const TransducerAssets& assets) {
    if (encoder_engine == nullptr) {
        return Status::Error(-6, "transducer: no encoder engine");
    }
    if (!assets.complete()) {
        return Status::Error(-6, "transducer: incomplete model assets");
    }
    encoder_ = encoder_engine;
    Status status = encoder_->LoadModelFromBuffer(assets.encoder.Bytes(), assets.encoder.Size());
    if (!status.ok()) return status;

    EpPreference cpu;
    cpu.ep_enum = VIETASR_EP_CPU;
    decoder_ = EngineFactory::CreateAuto(cpu);
    joiner_ = EngineFactory::CreateAuto(cpu);
    if (!decoder_ || !joiner_) {
        return Status::Error(-6, "transducer: cannot create decoder/joiner engines");
    }
    decoder_->SetIntraOpThreads(1);
    joiner_->SetIntraOpThreads(1);
    status = decoder_->LoadModelFromBuffer(assets.decoder.Bytes(), assets.decoder.Size());
    if (!status.ok()) return status;
    return joiner_->LoadModelFromBuffer(assets.joiner.Bytes(), assets.joiner.Size());
}

Status TransducerModel::Encode(const std::vector<float>& features, int num_frames,
                               EncoderOutput* out) const {
    std::vector<Tensor> inputs(2);
    inputs[0].shape = {1, num_frames, feature_dim()};
    inputs[0].data = features;
    inputs[1].shape = {1};
    inputs[1].data = {static_cast<float>(num_frames)};
    std::vector<Tensor> outputs;
    Status status = encoder_->Run(inputs, &outputs);
    if (!status.ok()) return status;
    if (outputs.size() < 2 || outputs[0].shape.size() != 3) {
        return Status::Error(-6, "transducer: unexpected encoder outputs");
    }
    out->dim = static_cast<int>(outputs[0].shape[2]);
    int produced = static_cast<int>(outputs[0].shape[1]);
    int valid = outputs[1].data.empty() ? produced : static_cast<int>(outputs[1].data[0]);
    out->frames = valid < produced ? valid : produced;
    out->data = std::move(outputs[0].data);
    out->data.resize(static_cast<std::size_t>(out->frames) * static_cast<std::size_t>(out->dim));
    return Status::Ok();
}

Status TransducerModel::Decode(const std::vector<std::int64_t>& contexts, int batch,
                               std::vector<float>* out) const {
    std::vector<Tensor> inputs(1);
    inputs[0].shape = {batch, context_size_};
    inputs[0].data.assign(contexts.begin(), contexts.end());
    std::vector<Tensor> outputs;
    Status status = decoder_->Run(inputs, &outputs);
    if (!status.ok()) return status;
    if (outputs.empty()) return Status::Error(-6, "transducer: decoder returned nothing");
    *out = std::move(outputs[0].data);
    return Status::Ok();
}

Status TransducerModel::Join(const std::vector<float>& encoder_rows,
                             const std::vector<float>& decoder_rows, int batch,
                             std::vector<float>* logits) const {
    int enc_dim = static_cast<int>(encoder_rows.size()) / batch;
    int dec_dim = static_cast<int>(decoder_rows.size()) / batch;
    std::vector<Tensor> inputs(2);
    inputs[0].shape = {batch, enc_dim};
    inputs[0].data = encoder_rows;
    inputs[1].shape = {batch, dec_dim};
    inputs[1].data = decoder_rows;
    std::vector<Tensor> outputs;
    Status status = joiner_->Run(inputs, &outputs);
    if (!status.ok()) return status;
    if (outputs.empty()) return Status::Error(-6, "transducer: joiner returned nothing");
    *logits = std::move(outputs[0].data);
    return Status::Ok();
}

}
