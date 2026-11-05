#ifndef VIETASR_MODULES_VIETASR_MODEL_FETCHER_H
#define VIETASR_MODULES_VIETASR_MODEL_FETCHER_H

#include <string>

#include "vietasr/model_manager.h"
#include "vietasr/types.h"

namespace vietasr {

class TransducerModelFetcher final {
public:
    explicit TransducerModelFetcher(ModelManager* models);

    static ModelBundle Bundle();
    static std::string BaseUrl();
    static std::string Revision();
    static bool OfflineByEnvironment();

    std::string CacheDir() const;
    Status Fetch(std::string* local_dir) const;

private:
    ModelManager* models_;
};

}

#endif
