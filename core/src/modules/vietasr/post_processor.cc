#include "modules/vietasr/post_processor.h"

#include <regex>
#include <string>

namespace vietasr {

std::string PostProcessor::Detokenize(const std::vector<std::string>& tokens) const {
    std::string out;
    out.reserve(tokens.size() * 4);
    for (const auto& token : tokens) {
        if (token.size() == 6 && token.compare(0, 3, "<0x") == 0 && token[5] == '>') {
            out.push_back(static_cast<char>(std::stoi(token.substr(3, 2), nullptr, 16)));
        } else if (token.rfind("\xe2\x96\x81", 0) == 0) {
            if (!out.empty()) out.push_back(' ');
            out.append(token, 3, std::string::npos);
        } else if (token == "<unk>" || token == "<blk>" || token == "<sos/eos>"
                   || token == "<context>" || token == "</context>") {
            continue;
        } else {
            out.append(token);
        }
    }
    return out;
}

std::string PostProcessor::StripBadWords(const std::string& text,
                                         const std::string& regex_pattern) const {
    if (regex_pattern.empty()) return text;
    try {
        std::regex pattern(regex_pattern);
        return std::regex_replace(text, pattern, "");
    } catch (const std::regex_error&) {
        return text;
    }
}

}
