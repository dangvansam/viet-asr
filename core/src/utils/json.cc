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

// --------------------------------------------------------------------------
// Flat-object scalar readers
// --------------------------------------------------------------------------

namespace {

bool IsWs(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

void SkipWs(const std::string& s, std::size_t* p) {
    while (*p < s.size() && IsWs(s[*p])) ++*p;
}

// Parse a JSON string starting at s[*p]=='"'; decode escapes into *out and
// advance *p past the closing quote. Returns false on malformed input.
bool ParseString(const std::string& s, std::size_t* p, std::string* out) {
    if (*p >= s.size() || s[*p] != '"') return false;
    ++*p;
    out->clear();
    while (*p < s.size()) {
        char c = s[(*p)++];
        if (c == '"') return true;
        if (c != '\\') {
            out->push_back(c);
            continue;
        }
        if (*p >= s.size()) return false;
        char e = s[(*p)++];
        switch (e) {
            case '"':  out->push_back('"');  break;
            case '\\': out->push_back('\\'); break;
            case '/':  out->push_back('/');  break;
            case 'b':  out->push_back('\b'); break;
            case 'f':  out->push_back('\f'); break;
            case 'n':  out->push_back('\n'); break;
            case 'r':  out->push_back('\r'); break;
            case 't':  out->push_back('\t'); break;
            case 'u': {
                if (*p + 4 > s.size()) return false;
                unsigned cp = 0;
                for (int i = 0; i < 4; ++i) {
                    char h = s[(*p)++];
                    cp <<= 4;
                    if (h >= '0' && h <= '9')      cp |= unsigned(h - '0');
                    else if (h >= 'a' && h <= 'f') cp |= unsigned(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') cp |= unsigned(h - 'A' + 10);
                    else return false;
                }
                // Encode the BMP code point as UTF-8 (surrogate pairs are not
                // needed for module config values).
                if (cp < 0x80) {
                    out->push_back(char(cp));
                } else if (cp < 0x800) {
                    out->push_back(char(0xC0 | (cp >> 6)));
                    out->push_back(char(0x80 | (cp & 0x3F)));
                } else {
                    out->push_back(char(0xE0 | (cp >> 12)));
                    out->push_back(char(0x80 | ((cp >> 6) & 0x3F)));
                    out->push_back(char(0x80 | (cp & 0x3F)));
                }
                break;
            }
            default: return false;
        }
    }
    return false;  // unterminated
}

// Advance *p past a JSON value (string / object / array / scalar literal).
bool SkipValue(const std::string& s, std::size_t* p) {
    SkipWs(s, p);
    if (*p >= s.size()) return false;
    char c = s[*p];
    if (c == '"') {
        std::string tmp;
        return ParseString(s, p, &tmp);
    }
    if (c == '{' || c == '[') {
        char open = c;
        int depth = 0;
        while (*p < s.size()) {
            char d = s[*p];
            if (d == '"') {
                std::string tmp;
                if (!ParseString(s, p, &tmp)) return false;
                continue;
            }
            ++*p;
            if (d == open) {
                ++depth;
            } else if ((open == '{' && d == '}') ||
                       (open == '[' && d == ']')) {
                --depth;
                if (depth == 0) return true;
            }
        }
        return false;
    }
    // Scalar literal: number / true / false / null — read up to a delimiter.
    while (*p < s.size()) {
        char d = s[*p];
        if (d == ',' || d == '}' || d == ']' || IsWs(d)) break;
        ++*p;
    }
    return true;
}

// Locate key in a flat JSON object. On success sets *is_string and *value
// (the decoded string, or the raw scalar token).
bool FindValue(const std::string& json, const std::string& key,
               bool* is_string, std::string* value) {
    std::size_t p = 0;
    SkipWs(json, &p);
    if (p >= json.size() || json[p] != '{') return false;
    ++p;
    while (true) {
        SkipWs(json, &p);
        if (p >= json.size() || json[p] == '}') return false;
        if (json[p] != '"') return false;
        std::string k;
        if (!ParseString(json, &p, &k)) return false;
        SkipWs(json, &p);
        if (p >= json.size() || json[p] != ':') return false;
        ++p;
        SkipWs(json, &p);
        if (p >= json.size()) return false;
        if (json[p] == '"') {
            std::string sv;
            if (!ParseString(json, &p, &sv)) return false;
            if (k == key) {
                *is_string = true;
                *value = sv;
                return true;
            }
        } else {
            std::size_t val_start = p;
            if (!SkipValue(json, &p)) return false;
            if (k == key) {
                std::string raw = json.substr(val_start, p - val_start);
                while (!raw.empty() && IsWs(raw.back())) raw.pop_back();
                *is_string = false;
                *value = raw;
                return true;
            }
        }
        SkipWs(json, &p);
        if (p < json.size() && json[p] == ',') {
            ++p;
            continue;
        }
        return false;
    }
}

}  // namespace

bool JsonGetString(const std::string& json, const std::string& key,
                   std::string* out) {
    bool is_string = false;
    std::string value;
    if (!FindValue(json, key, &is_string, &value)) return false;
    if (!is_string) return false;
    *out = value;
    return true;
}

bool JsonGetNumber(const std::string& json, const std::string& key,
                   double* out) {
    bool is_string = false;
    std::string value;
    if (!FindValue(json, key, &is_string, &value)) return false;
    try {
        std::size_t idx = 0;
        double d = std::stod(value, &idx);
        if (idx == 0) return false;
        *out = d;
        return true;
    } catch (...) {
        return false;
    }
}

bool JsonGetBool(const std::string& json, const std::string& key, bool* out) {
    bool is_string = false;
    std::string value;
    if (!FindValue(json, key, &is_string, &value)) return false;
    if (value == "true")  { *out = true;  return true; }
    if (value == "false") { *out = false; return true; }
    return false;
}

}
