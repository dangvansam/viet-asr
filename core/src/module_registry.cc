#include "vietasr/module.h"

#include <mutex>
#include <unordered_map>

namespace vietasr {

class ModuleRegistry::Impl {
public:
    mutable std::mutex mutex;
    std::unordered_map<std::string, ModuleFactory> factories;
};

ModuleRegistry::ModuleRegistry() : impl_(std::make_unique<Impl>()) {}
ModuleRegistry::~ModuleRegistry() = default;

ModuleRegistry& ModuleRegistry::Instance() {
    static ModuleRegistry instance;
    return instance;
}

void ModuleRegistry::Register(const std::string& name, ModuleFactory factory) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->factories[name] = std::move(factory);
}

std::unique_ptr<Module> ModuleRegistry::Create(const std::string& name) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->factories.find(name);
    if (it == impl_->factories.end()) return nullptr;
    return it->second();
}

std::vector<std::string> ModuleRegistry::List() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::vector<std::string> names;
    names.reserve(impl_->factories.size());
    for (const auto& [name, _] : impl_->factories) names.push_back(name);
    return names;
}

}
