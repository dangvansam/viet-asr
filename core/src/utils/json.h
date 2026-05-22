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

// Minimal flat-object scalar readers. Each extracts a top-level scalar by key
// from a flat JSON object string — the form module config arrives in (e.g.
// {"backend":"silero","threshold":0.6}). Nested objects/arrays are skipped, not
// descended into. Every reader returns false (leaving *out untouched) when the
// key is absent, the value has the wrong type, or the input is not an object.
bool JsonGetString(const std::string& json, const std::string& key,
                   std::string* out);
bool JsonGetNumber(const std::string& json, const std::string& key,
                   double* out);
bool JsonGetBool(const std::string& json, const std::string& key, bool* out);

}

#endif
