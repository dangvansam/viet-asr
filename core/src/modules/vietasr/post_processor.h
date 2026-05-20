#ifndef VIETASR_MODULES_VIETASR_POST_PROCESSOR_H
#define VIETASR_MODULES_VIETASR_POST_PROCESSOR_H

#include <string>
#include <vector>

namespace vietasr {

class PostProcessor final {
public:
    std::string Detokenize(const std::vector<std::string>& tokens) const;
    std::string StripBadWords(const std::string& text,
                              const std::string& regex_pattern) const;
};

}

#endif
