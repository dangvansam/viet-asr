#ifndef VIETASR_PIPELINE_H
#define VIETASR_PIPELINE_H

#include <memory>
#include <string>
#include <vector>

#include "vietasr/engine.h"
#include "vietasr/model_manager.h"
#include "vietasr/module.h"
#include "vietasr/result.h"
#include "vietasr/types.h"

namespace vietasr {

class Session;

class Pipeline final {
public:
    Pipeline();
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    Status AddModule(const std::string& name, const std::string& json_config);
    Status SetBackend(int backend_enum);
    Status SetModelDir(const std::string& model_dir);

    Status Build();
    Status EnsureModels();

    std::unique_ptr<Session> NewSession(float sample_rate);

    std::string TranscribeFile(const std::string& wav_path);
    std::string TranscribeBuffer(const short* pcm, int len, float sample_rate);

    static std::unique_ptr<Pipeline> FromPreset(const std::string& preset_name);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    friend class Session;
};

class Session final {
public:
    explicit Session(Pipeline::Impl* pipeline_impl, float sample_rate);
    ~Session();

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    void Reset();

    int AcceptWaveformS16(const short* pcm, int len);
    int AcceptWaveformF32(const float* pcm, int len);

    std::string Partial();
    std::string Result();
    std::string Final();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class PresetRegistry final {
public:
    using Builder = std::function<Status(Pipeline*)>;

    static PresetRegistry& Instance();
    void Register(const std::string& name, Builder builder);
    Status Apply(const std::string& name, Pipeline* pipeline) const;
    std::vector<std::string> List() const;

private:
    PresetRegistry();
    ~PresetRegistry();
    class Impl;
    std::unique_ptr<Impl> impl_;
};

#define VIETASR_REGISTER_PRESET(name, builder_fn)                             \
    namespace {                                                               \
    struct AutoRegisterPreset_##builder_fn {                                  \
        AutoRegisterPreset_##builder_fn() {                                   \
            ::vietasr::PresetRegistry::Instance().Register(                   \
                name, builder_fn);                                            \
        }                                                                     \
    };                                                                        \
    static AutoRegisterPreset_##builder_fn s_preset_##builder_fn;             \
    }

}

#endif
