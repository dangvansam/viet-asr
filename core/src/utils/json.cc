#include "utils/json.h"

#include <cmath>
#include <cstdio>

namespace vietasr {

std::string JsonEscape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    out.push_back('"');
    for (char raw : text) {
        unsigned char byte = static_cast<unsigned char>(raw);
        switch (byte) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (byte < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", byte);
                    out += buf;
                } else {
                    out.push_back(raw);
                }
        }
    }
    out.push_back('"');
    return out;
}

void JsonWriter::HandleComma() {
    if (!needs_comma_.empty() && needs_comma_.back()) {
        stream_ << ',';
    }
    if (!needs_comma_.empty()) {
        needs_comma_.back() = true;
    }
}

JsonWriter& JsonWriter::BeginObject() {
    HandleComma();
    stream_ << '{';
    needs_comma_.push_back(false);
    return *this;
}

JsonWriter& JsonWriter::EndObject() {
    stream_ << '}';
    if (!needs_comma_.empty()) needs_comma_.pop_back();
    return *this;
}

JsonWriter& JsonWriter::BeginArray() {
    HandleComma();
    stream_ << '[';
    needs_comma_.push_back(false);
    return *this;
}

JsonWriter& JsonWriter::EndArray() {
    stream_ << ']';
    if (!needs_comma_.empty()) needs_comma_.pop_back();
    return *this;
}

JsonWriter& JsonWriter::Key(const std::string& key) {
    HandleComma();
    stream_ << JsonEscape(key) << ':';
    if (!needs_comma_.empty()) {
        needs_comma_.back() = false;
    }
    return *this;
}

JsonWriter& JsonWriter::String(const std::string& value) {
    HandleComma();
    stream_ << JsonEscape(value);
    return *this;
}

JsonWriter& JsonWriter::RawValue(const std::string& json_literal) {
    HandleComma();
    stream_ << json_literal;
    return *this;
}

JsonWriter& JsonWriter::Number(double value) {
    HandleComma();
    if (std::isnan(value) || std::isinf(value)) {
        stream_ << "null";
    } else {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%g", value);
        stream_ << buf;
    }
    return *this;
}

JsonWriter& JsonWriter::Integer(long long value) {
    HandleComma();
    stream_ << value;
    return *this;
}

JsonWriter& JsonWriter::Boolean(bool value) {
    HandleComma();
    stream_ << (value ? "true" : "false");
    return *this;
}

JsonWriter& JsonWriter::Null() {
    HandleComma();
    stream_ << "null";
    return *this;
}

std::string JsonWriter::Str() const {
    return stream_.str();
}

}
