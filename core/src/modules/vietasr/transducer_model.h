#ifndef VIETASR_MODULES_VIETASR_TRANSDUCER_MODEL_H
#define VIETASR_MODULES_VIETASR_TRANSDUCER_MODEL_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vietasr/engine.h"
#include "vietasr/types.h"

namespace vietasr {

struct ModelBlob final {
    const unsigned char* data{nullptr};
    std::size_t size{0};
    std::vector<unsigned char> owned;

    bool empty() const { return Bytes() == nullptr || Size() == 0; }
    const unsigned char* Bytes() const { return owned.empty() ? data : owned.data(); }
    std::size_t Size() const { return owned.empty() ? size : owned.size(); }
};

struct TransducerAssets final {
    ModelBlob encoder;
    ModelBlob decoder;
    ModelBlob joiner;
    std::string tokens;
    std::string origin;

    bool complete() const {
        return !encoder.empty() && !decoder.empty() && !joiner.empty() && !tokens.empty();
    }
};

class TransducerAssetLoader final {
public:
    static TransducerAssets FromEmbedded();
    static TransducerAssets FromDirectory(const std::string& dir);
    static TransducerAssets Resolve(const std::string& configured_dir,
                                    const std::string& cache_dir);

private:
    static bool ReadFile(const std::string& path, std::vector<unsigned char>* out);
    static std::string FirstExisting(const std::string& dir,
                                     const std::vector<std::string>& names);
};

struct EncoderOutput final {
    std::vector<float> data;
    int frames{0};
    int dim{0};

    const float* Frame(int t) const {
        return data.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(dim);
    }
};

class TransducerModel final {
public:
    TransducerModel();
    ~TransducerModel();

    Status Load(Engine* encoder_engine, const TransducerAssets& assets);

    Status Encode(const std::vector<float>& features, int num_frames,
                  EncoderOutput* out) const;
    Status Decode(const std::vector<std::int64_t>& contexts, int batch,
                  std::vector<float>* out) const;
    Status Join(const std::vector<float>& encoder_rows,
                const std::vector<float>& decoder_rows, int batch,
                std::vector<float>* logits) const;

    int context_size() const { return context_size_; }
    int feature_dim() const { return 80; }
    int blank_id() const { return 0; }

private:
    Engine* encoder_{nullptr};
    std::unique_ptr<Engine> decoder_;
    std::unique_ptr<Engine> joiner_;
    int context_size_{2};
};

}

#endif
