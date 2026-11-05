#include "modules/vietasr/model_fetcher.h"

#include <cstdlib>

#include "utils/fs.h"
#include "vietasr/logger.h"
#include "vietasr_model_source.h"

namespace vietasr {

namespace {

constexpr const char* kModule = "vietasr";

std::string EnvValue(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
}

}

TransducerModelFetcher::TransducerModelFetcher(ModelManager* models) : models_(models) {}

std::string TransducerModelFetcher::BaseUrl() {
    std::string url = EnvValue("VIETASR_MODEL_BASE_URL");
    if (url.empty()) url = model_source::kBaseUrl;
    if (!url.empty() && url.back() != '/') url.push_back('/');
    return url;
}

std::string TransducerModelFetcher::Revision() {
    return std::string(model_source::kRevision).substr(0, 12);
}

bool TransducerModelFetcher::OfflineByEnvironment() {
    std::string value = EnvValue("VIETASR_OFFLINE");
    return !value.empty() && value != "0" && value != "false" && value != "FALSE";
}

ModelBundle TransducerModelFetcher::Bundle() {
    ModelBundle bundle;
    bundle.module = kModule;
    bundle.version = Revision();
    std::string base = BaseUrl();
    for (std::size_t i = 0; i < model_source::kFileCount; ++i) {
        const model_source::File& file = model_source::kFiles[i];
        ModelFile entry;
        entry.name = file.name;
        entry.url = base + file.name;
        entry.md5 = file.md5;
        entry.size_bytes = static_cast<std::int64_t>(file.size);
        bundle.files.push_back(entry);
    }
    return bundle;
}

std::string TransducerModelFetcher::CacheDir() const {
    if (models_ == nullptr || models_->cache_dir().empty()) return std::string();
    return FileSystem::JoinPath(FileSystem::JoinPath(models_->cache_dir(), kModule), Revision());
}

Status TransducerModelFetcher::Fetch(std::string* local_dir) const {
    if (models_ == nullptr) return Status::Error(-6, "vietasr: no model manager");
    ModelBundle bundle = Bundle();
    VIETASR_LOG_INFO("vietasr") << "downloading model " << model_source::kRepo << "@" << bundle.version
        << " from " << BaseUrl() << " into " << CacheDir();
    return models_->EnsureBundle(bundle, local_dir);
}

}
