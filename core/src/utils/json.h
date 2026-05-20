#ifndef VIETASR_UTILS_JSON_H
#define VIETASR_UTILS_JSON_H

#include <sstream>
#include <string>
#include <vector>

namespace vietasr {

class JsonWriter final {
public:
    JsonWriter& BeginObject();
    JsonWriter& EndObject();
    JsonWriter& BeginArray();
    JsonWriter& EndArray();
    JsonWriter& Key(const std::string& key);
    JsonWriter& String(const std::string& value);
    JsonWriter& RawValue(const std::string& json_literal);
    JsonWriter& Number(double value);
    JsonWriter& Integer(long long value);
    JsonWriter& Boolean(bool value);
    JsonWriter& Null();

    std::string Str() const;

private:
    std::ostringstream stream_;
    std::vector<bool> needs_comma_;

    void HandleComma();
};

std::string JsonEscape(const std::string& text);

}

#endif
