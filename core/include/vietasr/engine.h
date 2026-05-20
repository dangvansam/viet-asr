#ifndef VIETASR_ENGINE_H
#define VIETASR_ENGINE_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "vietasr/types.h"

namespace vietasr {

struct Tensor final {
    std::vector<float> data;
    std::vector<std::int64_t> shape;
};

class Engine {
public:
    virtual ~Engine() = default;

    virtual const char* backend_name() const = 0;

    virtual Status LoadModel(const std::string& model_path) = 0;

    virtual Status LoadModelFromBuffer(const void* data, std::size_t size) {
        (void)data;
        (void)size;
        return Status::Error(-6,
            "engine: LoadModelFromBuffer not supported by this backend");
    }

    virtual Status Run(const std::vector<Tensor>& inputs,
                       std::vector<Tensor>* outputs) = 0;

    virtual void Reset() {}
};

class EngineFactory final {
public:
    static std::unique_ptr<Engine> Create(int backend_enum);
    static std::unique_ptr<Engine> CreateAuto();
};

}

#endif
