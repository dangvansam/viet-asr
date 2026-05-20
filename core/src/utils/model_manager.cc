#include "vietasr/model_manager.h"

#include <mutex>
#include <unordered_map>

#include "utils/fs.h"
#include "utils/http_client.h"
#include "utils/md5.h"
#include "vietasr/logger.h"

namespace vietasr {

class ModelManager::Impl {
public:
    std::mutex mutex;
    std::string cache_dir;
    std::string manifest_url{"https://cdn.vietasr.io/manifest.json"};
    std::unordered_map<std::string, ModelBundle> bundles;
    HttpClient http;
};

ModelManager::ModelManager() : impl_(std::make_unique<Impl>()) {
    impl_->cache_dir = FileSystem::DefaultCacheDir();
}

ModelManager::~ModelManager() = default;

void ModelManager::SetCacheDir(const std::string& cache_dir) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->cache_dir = cache_dir;
}

const std::string& ModelManager::cache_dir() const {
    return impl_->cache_dir;
}

void ModelManager::SetManifestUrl(const std::string& manifest_url) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->manifest_url = manifest_url;
}

Status ModelManager::EnsureBundle(const ModelBundle& bundle, std::string* local_dir) {
    std::string dir = FileSystem::JoinPath(
        FileSystem::JoinPath(impl_->cache_dir, bundle.module),
        bundle.version);
    if (!FileSystem::EnsureDirectory(dir)) {
        return Status::Error(-3, "failed to create cache directory: " + dir);
    }

    for (const auto& file : bundle.files) {
        std::string path = FileSystem::JoinPath(dir, file.name);
        if (FileSystem::FileExists(path)) {
            std::string actual = Md5::HashFile(path);
            if (!file.md5.empty() && actual != file.md5
                && file.md5.rfind("TODO", 0) != 0) {
                VIETASR_LOG_WARN("model_manager")
                    << "checksum mismatch for " << file.name
                    << " (expected " << file.md5 << ", got " << actual
                    << "); will re-download";
            } else {
                continue;
            }
        }
        auto status = impl_->http.Download(file.url, path);
        if (!status.ok()) {
            return Status::Error(status.code(),
                "download failed: " + file.url + " (" + status.message() + ")");
        }
        if (!file.md5.empty() && file.md5.rfind("TODO", 0) != 0) {
            std::string actual = Md5::HashFile(path);
            if (actual != file.md5) {
                return Status::Error(-5,
                    "checksum mismatch after download: " + file.name);
            }
        }
    }

    if (local_dir) *local_dir = dir;
    return Status::Ok();
}

Status ModelManager::LoadManifest() {
    std::vector<unsigned char> body;
    auto status = impl_->http.Get(impl_->manifest_url, &body);
    if (!status.ok()) {
        return Status::Error(status.code(),
            "manifest unavailable: " + status.message());
    }
    return Status::Ok();
}

Status ModelManager::LookupBundle(const std::string& module, ModelBundle* out) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->bundles.find(module);
    if (it == impl_->bundles.end()) {
        return Status::Error(-2, "no manifest entry for module: " + module);
    }
    *out = it->second;
    return Status::Ok();
}

std::string ModelManager::DefaultCacheDir() {
    return FileSystem::DefaultCacheDir();
}

}
