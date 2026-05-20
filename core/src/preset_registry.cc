#include "vietasr/pipeline.h"

#include <mutex>
#include <unordered_map>

namespace vietasr {

class PresetRegistry::Impl {
public:
    mutable std::mutex mutex;
    std::unordered_map<std::string, Builder> builders;
};

PresetRegistry::PresetRegistry() : impl_(std::make_unique<Impl>()) {}
PresetRegistry::~PresetRegistry() = default;

PresetRegistry& PresetRegistry::Instance() {
    static PresetRegistry instance;
    return instance;
}

void PresetRegistry::Register(const std::string& name, Builder builder) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->builders[name] = std::move(builder);
}

Status PresetRegistry::Apply(const std::string& name, Pipeline* pipeline) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->builders.find(name);
    if (it == impl_->builders.end()) {
        return Status::Error(-2, "unknown preset: " + name);
    }
    auto builder = it->second;
    return builder(pipeline);
}

std::vector<std::string> PresetRegistry::List() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::vector<std::string> names;
    names.reserve(impl_->builders.size());
    for (const auto& [name, _] : impl_->builders) names.push_back(name);
    return names;
}

}
