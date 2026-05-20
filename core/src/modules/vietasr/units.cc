#include "modules/vietasr/units.h"

#include <fstream>
#include <sstream>

namespace vietasr {

namespace {

void ParseUnitsStream(std::istream& stream, std::vector<std::string>* tokens) {
    tokens->clear();
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::size_t space = line.find(' ');
        std::string token = (space == std::string::npos) ? line : line.substr(0, space);
        tokens->push_back(std::move(token));
    }
}

}

Status Units::Load(const std::string& units_txt_path) {
    std::ifstream stream(units_txt_path);
    if (!stream) {
        return Status::Error(-3, "cannot open units file: " + units_txt_path);
    }
    ParseUnitsStream(stream, &tokens_);
    return Status::Ok();
}

Status Units::LoadFromText(const std::string& units_text) {
    std::istringstream stream(units_text);
    ParseUnitsStream(stream, &tokens_);
    if (tokens_.empty()) {
        return Status::Error(-3, "units text is empty");
    }
    return Status::Ok();
}

const std::string& Units::At(int token_id) const {
    if (token_id < 0 || token_id >= static_cast<int>(tokens_.size())) {
        return empty_;
    }
    return tokens_[token_id];
}

}
