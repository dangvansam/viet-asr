#ifndef VIETASR_MODULES_VIETASR_UNITS_H
#define VIETASR_MODULES_VIETASR_UNITS_H

#include <string>
#include <vector>

#include "vietasr/types.h"

namespace vietasr {

class Units final {
public:
    Status Load(const std::string& units_txt_path);
    Status LoadFromText(const std::string& units_text);
    const std::string& At(int token_id) const;
    int size() const { return static_cast<int>(tokens_.size()); }

private:
    std::vector<std::string> tokens_;
    std::string empty_;
};

}

#endif
