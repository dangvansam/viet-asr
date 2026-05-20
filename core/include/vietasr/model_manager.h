#ifndef VIETASR_MODEL_MANAGER_H
#define VIETASR_MODEL_MANAGER_H

#include <memory>
#include <string>
#include <vector>

#include "vietasr/types.h"

namespace vietasr {

struct ModelFile final {
    std::string name;
    std::string url;
    std::string md5;
    std::int64_t size_bytes;
};

struct ModelBundle final {
    std::string module;
    std::string version;
    std::vector<ModelFile> files;
};

class ModelManager final {
public:
    ModelManager();
    ~ModelManager();

    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;

    void SetCacheDir(const std::string& cache_dir);
    const std::string& cache_dir() const;

    void SetManifestUrl(const std::string& manifest_url);

    Status EnsureBundle(const ModelBundle& bundle, std::string* local_dir);
    Status LoadManifest();
    Status LookupBundle(const std::string& module, ModelBundle* out) const;

    static std::string DefaultCacheDir();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}

#endif
