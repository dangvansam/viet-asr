#include "vietasr/pipeline.h"

#include <mutex>
#include <utility>
#include <vector>

#include "preprocess/audio_io.h"
#include "preprocess/audio_resampler.h"
#include "utils/json.h"
#include "vietasr/logger.h"

namespace vietasr {

class Pipeline::Impl {
public:
    struct Entry {
        std::string name;
        std::string config_json;
        std::unique_ptr<Module> module;
    };

    std::vector<Entry> entries;
    std::unique_ptr<Engine> engine;
    std::unique_ptr<ModelManager> models;
    int backend{0};
    std::string model_dir;
    bool built{false};
    std::mutex drive_mutex;

    Impl() : models(std::make_unique<ModelManager>()) {}

    void DriveBatch(const AudioClip& clip, ResultBuilder* out) {
        for (auto& entry : entries) {
            entry.module->Reset();
        }

        constexpr int kTargetRate = 16000;
        AudioClip resampled;
        const AudioClip* effective = &clip;
        if (std::abs(clip.sample_rate - static_cast<float>(kTargetRate)) > 0.5f) {
            AudioResampler resampler(static_cast<int>(clip.sample_rate), kTargetRate);
            resampler.AcceptWaveform(clip.pcm.data(), clip.pcm.size());
            resampler.Flush();
            resampled.pcm = resampler.PopAll();
            resampled.sample_rate = static_cast<float>(kTargetRate);
            VIETASR_LOG_DEBUG("pipeline")
                << "resampled " << clip.pcm.size() << " @ " << clip.sample_rate
                << " Hz -> " << resampled.pcm.size() << " @ " << kTargetRate << " Hz";
            effective = &resampled;
        }

        AudioFrame frame{
            effective->pcm.data(),
            effective->pcm.size(),
            effective->sample_rate,
            0
        };
        for (auto& entry : entries) {
            entry.module->OnFrame(frame, out);
        }

        Segment full_segment{
            0.0,
            clip.pcm.size() / static_cast<double>(clip.sample_rate),
            0,
            static_cast<std::int64_t>(clip.pcm.size()),
            true,
            true
        };
        for (auto& entry : entries) {
            entry.module->OnSegment(full_segment, out);
        }

        for (auto& entry : entries) {
            entry.module->OnFinalize(out);
        }

        out->MarkFinal(true);
    }

    void UpdateFullSegmentSampleRate(Segment* segment, float sample_rate, std::size_t n) {
        segment->end_s = static_cast<double>(n) / static_cast<double>(sample_rate);
        segment->end_sample = static_cast<std::int64_t>(n);
    }
};

Pipeline::Pipeline() : impl_(std::make_unique<Impl>()) {}
Pipeline::~Pipeline() = default;

Status Pipeline::AddModule(const std::string& name, const std::string& json_config) {
    auto module = ModuleRegistry::Instance().Create(name);
    if (!module) {
        return Status::Error(-2, "unknown module: " + name);
    }
    impl_->entries.push_back({name, json_config, std::move(module)});
    return Status::Ok();
}

Status Pipeline::SetBackend(int backend_enum) {
    impl_->backend = backend_enum;
    return Status::Ok();
}

Status Pipeline::SetModelDir(const std::string& model_dir) {
    impl_->model_dir = model_dir;
    impl_->models->SetCacheDir(model_dir);
    return Status::Ok();
}

Status Pipeline::EnsureModels() {
    for (auto& entry : impl_->entries) {
        ModelBundle bundle;
        if (impl_->models->LookupBundle(entry.name, &bundle).ok()) {
            std::string local;
            auto status = impl_->models->EnsureBundle(bundle, &local);
            if (!status.ok()) {
                VIETASR_LOG_WARN("pipeline")
                    << "EnsureBundle failed for " << entry.name
                    << ": " << status.message();
            }
        }
    }
    return Status::Ok();
}

Status Pipeline::Build() {
    impl_->engine = EngineFactory::Create(impl_->backend);
    EnsureModels();

    ModuleConfig config;
    config.sample_rate = 16000.0f;
    for (auto& entry : impl_->entries) {
        config.json = entry.config_json;
        auto status = entry.module->Init(config, impl_->models.get(), impl_->engine.get());
        if (!status.ok()) {
            return Status::Error(status.code(),
                "module init failed [" + entry.name + "]: " + status.message());
        }
    }
    impl_->built = true;
    return Status::Ok();
}

std::unique_ptr<Session> Pipeline::NewSession(float sample_rate) {
    if (!impl_->built) {
        auto status = Build();
        if (!status.ok()) return nullptr;
    }
    return std::make_unique<Session>(impl_.get(), sample_rate);
}

std::string Pipeline::TranscribeFile(const std::string& wav_path) {
    if (!impl_->built) {
        auto status = Build();
        if (!status.ok()) {
            ResultBuilder builder;
            builder.SetField("error", JsonEscape(status.message()));
            return builder.FinalJson();
        }
    }
    AudioClip clip;
    auto status = AudioIo::ReadWav(wav_path, &clip);
    if (!status.ok()) {
        ResultBuilder builder;
        builder.SetField("error", JsonEscape(status.message()));
        return builder.FinalJson();
    }
    auto session = NewSession(clip.sample_rate);
    if (!session) {
        ResultBuilder builder;
        builder.SetField("error", JsonEscape(std::string("session creation failed")));
        return builder.FinalJson();
    }
    session->AcceptWaveformF32(clip.pcm.data(), static_cast<int>(clip.pcm.size()));
    return session->Final();
}

std::string Pipeline::TranscribeBuffer(const short* pcm, int len, float sample_rate) {
    if (!impl_->built) {
        auto status = Build();
        if (!status.ok()) {
            ResultBuilder builder;
            builder.SetField("error", JsonEscape(status.message()));
            return builder.FinalJson();
        }
    }
    auto session = NewSession(sample_rate);
    if (!session) {
        ResultBuilder builder;
        builder.SetField("error", JsonEscape(std::string("session creation failed")));
        return builder.FinalJson();
    }
    session->AcceptWaveformS16(pcm, len);
    return session->Final();
}

std::unique_ptr<Pipeline> Pipeline::FromPreset(const std::string& preset_name) {
    auto pipeline = std::make_unique<Pipeline>();
    auto status = PresetRegistry::Instance().Apply(preset_name, pipeline.get());
    if (!status.ok()) {
        VIETASR_LOG_ERROR("pipeline") << "preset failed: " << status.message();
        return nullptr;
    }
    auto build_status = pipeline->Build();
    if (!build_status.ok()) {
        VIETASR_LOG_ERROR("pipeline")
            << "preset build failed: " << build_status.message();
        return nullptr;
    }
    return pipeline;
}

class Session::Impl {
public:
    Pipeline::Impl* pipeline;
    float input_sample_rate;
    float internal_sample_rate{16000.0f};
    ResultBuilder builder;
    std::vector<float> pending;
    bool finalized{false};
    std::unique_ptr<AudioResampler> resampler;
    std::int64_t internal_offset{0};
    std::vector<std::unique_ptr<Module>> modules;
    std::mutex mutex;

    void EnsureResampler() {
        if (resampler) return;
        if (std::abs(input_sample_rate - internal_sample_rate) <= 0.5f) return;
        resampler = std::make_unique<AudioResampler>(
            static_cast<int>(input_sample_rate),
            static_cast<int>(internal_sample_rate));
    }

    void FeedFrame(const float* data, std::size_t len) {
        AudioFrame frame{
            data, len, internal_sample_rate, internal_offset
        };
        for (auto& module : modules) {
            module->OnFrame(frame, &builder);
        }
        internal_offset += static_cast<std::int64_t>(len);
    }
};

Session::Session(Pipeline::Impl* pipeline_impl, float sample_rate)
    : impl_(std::make_unique<Impl>()) {
    impl_->pipeline = pipeline_impl;
    impl_->input_sample_rate = sample_rate;
    impl_->EnsureResampler();

    impl_->modules.reserve(pipeline_impl->entries.size());
    for (auto& entry : pipeline_impl->entries) {
        auto cloned = entry.module->Clone();
        if (!cloned) {
            VIETASR_LOG_WARN("session")
                << "module '" << entry.name
                << "' has no Clone(); falling back to shared instance "
                   "(this Session will not be safe for concurrent use)";
            continue;
        }
        impl_->modules.push_back(std::move(cloned));
    }
}

Session::~Session() = default;

void Session::Reset() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->builder.Reset();
    impl_->pending.clear();
    impl_->finalized = false;
    impl_->internal_offset = 0;
    if (impl_->resampler) impl_->resampler->Reset();
    for (auto& module : impl_->modules) {
        module->Reset();
    }
}

int Session::AcceptWaveformS16(const short* pcm, int len) {
    std::vector<float> chunk(len);
    for (int i = 0; i < len; ++i) {
        chunk[i] = static_cast<float>(pcm[i]) / 32768.0f;
    }
    return AcceptWaveformF32(chunk.data(), len);
}

int Session::AcceptWaveformF32(const float* pcm, int len) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->resampler) {
        impl_->resampler->AcceptWaveform(pcm, static_cast<std::size_t>(len));
        std::vector<float> resampled = impl_->resampler->PopAll();
        if (!resampled.empty()) {
            impl_->FeedFrame(resampled.data(), resampled.size());
        }
    } else {
        impl_->FeedFrame(pcm, static_cast<std::size_t>(len));
    }
    return 0;
}

std::string Session::Partial() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->builder.SnapshotJson();
}

std::string Session::Result() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->builder.SnapshotJson();
}

std::string Session::Final() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->finalized) {
        if (impl_->resampler) {
            impl_->resampler->Flush();
            std::vector<float> tail = impl_->resampler->PopAll();
            if (!tail.empty()) {
                impl_->FeedFrame(tail.data(), tail.size());
            }
        }
        for (auto& module : impl_->modules) {
            module->OnFinalize(&impl_->builder);
        }
        impl_->finalized = true;
    }
    return impl_->builder.FinalJson();
}

}
