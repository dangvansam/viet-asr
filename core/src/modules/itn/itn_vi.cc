#include "modules/itn/itn_vi.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Rule-based Vietnamese inverse text normalization. The transcript reaching
// this code is space-separated lowercase UTF-8 (the BPE detokenizer output).
// The algorithm: tokenize -> classify -> disambiguate -> segment into number
// runs -> evaluate each run to digits -> reassemble. It never touches per-byte
// UTF-8 internals — every decision is a whole-token map lookup — so it is
// UTF-8 safe without any locale/ICU dependency.

namespace vietasr {
namespace {

enum class Cat {
    kWord,      // not a number token — passthrough
    kUnit,      // a plain digit 0-9
    kVariant,   // positional digit variant: tư=4, lăm=5, mốt=1
    kTen,       // "mười" (10..19)
    kTenMult,   // "mươi" (tens multiplier, follows a digit)
    kScale,     // "trăm"/"nghìn"/"triệu"/"tỷ"
    kZero,      // "linh"/"lẻ" (a zero in the tens place)
    kDecimal,   // "phẩy"/"chấm" (decimal point)
};

struct Tok {
    std::string text;        // original word, emitted verbatim on passthrough
    Cat cat = Cat::kWord;
    int value = 0;           // digit value for kUnit/kVariant
    std::int64_t scale = 1;  // multiplier for kScale
};

struct LexEntry {
    Cat cat;
    int value;
    std::int64_t scale;
};

// Vietnamese number lexicon. Built once. Includes common spelling variants
// (bảy/bẩy, nghìn/ngàn, tỷ/tỉ).
const std::unordered_map<std::string, LexEntry>& Lexicon() {
    static const std::unordered_map<std::string, LexEntry> kLex = {
        {"không", {Cat::kUnit, 0, 1}},
        {"một", {Cat::kUnit, 1, 1}},
        {"hai", {Cat::kUnit, 2, 1}},
        {"ba", {Cat::kUnit, 3, 1}},
        {"bốn", {Cat::kUnit, 4, 1}},
        {"năm", {Cat::kUnit, 5, 1}},
        {"sáu", {Cat::kUnit, 6, 1}},
        {"bảy", {Cat::kUnit, 7, 1}},
        {"bẩy", {Cat::kUnit, 7, 1}},
        {"tám", {Cat::kUnit, 8, 1}},
        {"chín", {Cat::kUnit, 9, 1}},
        {"tư", {Cat::kVariant, 4, 1}},
        {"lăm", {Cat::kVariant, 5, 1}},
        {"mốt", {Cat::kVariant, 1, 1}},
        {"mười", {Cat::kTen, 10, 1}},
        {"mươi", {Cat::kTenMult, 0, 1}},
        {"trăm", {Cat::kScale, 0, 100}},
        {"nghìn", {Cat::kScale, 0, 1000}},
        {"ngàn", {Cat::kScale, 0, 1000}},
        {"triệu", {Cat::kScale, 0, 1000000}},
        {"tỷ", {Cat::kScale, 0, 1000000000}},
        {"tỉ", {Cat::kScale, 0, 1000000000}},
        {"linh", {Cat::kZero, 0, 1}},
        {"lẻ", {Cat::kZero, 0, 1}},
        {"phẩy", {Cat::kDecimal, 0, 1}},
        {"chấm", {Cat::kDecimal, 0, 1}},
    };
    return kLex;
}

bool IsNumberCat(Cat c) {
    return c == Cat::kUnit || c == Cat::kVariant || c == Cat::kTen ||
           c == Cat::kTenMult || c == Cat::kScale || c == Cat::kZero;
}

// kTen/kTenMult/kScale/kZero force numeral folding; a run of only these plus
// plain digits is a digit sequence and is concatenated instead.
bool IsStructCat(Cat c) {
    return c == Cat::kTen || c == Cat::kTenMult || c == Cat::kScale ||
           c == Cat::kZero;
}

// A single-word run is only folded when the word is a digit with no common
// homophone. Excludes không (no/not), ba (dad), năm (year), chín (ripe) and the
// positional variants — folding those in isolation corrupts ordinary speech.
bool IsSafeSoloUnit(const std::string& w) {
    return w == "một" || w == "hai" || w == "bốn" || w == "sáu" ||
           w == "bảy" || w == "bẩy" || w == "tám";
}

std::vector<std::string> Split(const std::string& text) {
    std::vector<std::string> words;
    std::string cur;
    for (char ch : text) {
        if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
            if (!cur.empty()) {
                words.push_back(cur);
                cur.clear();
            }
        } else {
            cur.push_back(ch);
        }
    }
    if (!cur.empty()) words.push_back(cur);
    return words;
}

// Evaluate a contiguous numeral segment [s, e) (no decimal tokens) to an
// integer. Standard trăm/nghìn/triệu/tỷ fold with two accumulators.
std::int64_t FoldNumeral(const std::vector<Tok>& t, int s, int e) {
    std::int64_t result = 0;   // accumulated across scale words
    std::int64_t current = 0;  // value within the current scale group
    int last_digit = 0;
    bool have_digit = false;
    for (int k = s; k < e; ++k) {
        const Tok& tok = t[k];
        switch (tok.cat) {
            case Cat::kUnit:
            case Cat::kVariant:
                // Added as a ones digit by default; a following mươi/trăm
                // promotes it to the tens/hundreds place.
                current += tok.value;
                last_digit = tok.value;
                have_digit = true;
                break;
            case Cat::kTen:  // "mười" => 10
                current += 10;
                have_digit = false;
                break;
            case Cat::kTenMult:  // "mươi" => the last digit was tens
                if (have_digit) {
                    current += static_cast<std::int64_t>(last_digit) * 9;
                }
                have_digit = false;
                break;
            case Cat::kScale:
                if (tok.scale == 100) {  // "trăm"
                    if (have_digit) {
                        current += static_cast<std::int64_t>(last_digit) * 99;
                    } else {
                        current += 100;
                    }
                    have_digit = false;
                } else {  // "nghìn"/"triệu"/"tỷ"
                    result += current * tok.scale;
                    current = 0;
                    have_digit = false;
                }
                break;
            case Cat::kZero:  // "linh"/"lẻ" — forces the next digit into ones
                have_digit = false;
                break;
            default:
                break;
        }
    }
    return result + current;
}

// Concatenate a segment of plain digits, e.g. "một chín chín tám" -> "1998".
// Preserves leading zeros (phone numbers, codes, fractional parts).
std::string ConcatDigits(const std::vector<Tok>& t, int s, int e) {
    std::string out;
    for (int k = s; k < e; ++k) {
        if (t[k].cat == Cat::kUnit || t[k].cat == Cat::kVariant) {
            out.push_back(static_cast<char>('0' + t[k].value));
        }
    }
    return out;
}

std::string SegmentToStr(const std::vector<Tok>& t, int s, int e) {
    if (s >= e) return std::string();
    for (int k = s; k < e; ++k) {
        if (IsStructCat(t[k].cat)) {
            return std::to_string(FoldNumeral(t, s, e));
        }
    }
    return ConcatDigits(t, s, e);
}

// Convert an accepted number run [lo, hi) to its digit-string form. Splits at
// decimal tokens; returns empty if the run cannot be evaluated cleanly.
std::string EvalRun(const std::vector<Tok>& t, int lo, int hi) {
    std::vector<std::pair<int, int>> segs;
    int seg_start = lo;
    for (int k = lo; k < hi; ++k) {
        if (t[k].cat == Cat::kDecimal) {
            segs.emplace_back(seg_start, k);
            seg_start = k + 1;
        }
    }
    segs.emplace_back(seg_start, hi);

    std::string out;
    for (std::size_t i = 0; i < segs.size(); ++i) {
        if (segs[i].first >= segs[i].second) return std::string();  // empty
        if (i > 0) out.push_back('.');
        out += SegmentToStr(t, segs[i].first, segs[i].second);
    }
    return out;
}

}  // namespace

std::string ApplyItnVietnamese(const std::string& text) {
    if (text.empty()) return text;
    std::vector<std::string> words = Split(text);
    if (words.empty()) return text;

    const auto& lex = Lexicon();
    std::vector<Tok> toks;
    toks.reserve(words.size());
    for (auto& w : words) {
        Tok tok;
        tok.text = w;
        auto it = lex.find(w);
        if (it != lex.end()) {
            tok.cat = it->second.cat;
            tok.value = it->second.value;
            tok.scale = it->second.scale;
        }
        toks.push_back(std::move(tok));
    }
    const int n = static_cast<int>(toks.size());

    // Disambiguate "năm" (digit 5 vs the word "year"): keep it as a number only
    // inside a numeric context — a neighbouring number word, or a decimal point
    // that bridges to one ("ba phẩy năm"). A lone "năm" between ordinary words
    // ("năm nay", "năm ngoái") stays as text.
    std::vector<bool> num_ctx(n);
    for (int i = 0; i < n; ++i) {
        num_ctx[i] = IsNumberCat(toks[i].cat) || toks[i].cat == Cat::kDecimal;
    }
    for (int i = 0; i < n; ++i) {
        if (toks[i].cat == Cat::kUnit && toks[i].text == "năm") {
            bool prev = i > 0 && num_ctx[i - 1];
            bool next = i + 1 < n && num_ctx[i + 1];
            if (!prev && !next) toks[i].cat = Cat::kWord;
        }
    }

    std::string out;
    out.reserve(text.size() + 8);
    bool first_out = true;
    auto emit = [&](const std::string& s) {
        if (!first_out) out.push_back(' ');
        out += s;
        first_out = false;
    };

    int i = 0;
    while (i < n) {
        if (!IsNumberCat(toks[i].cat)) {
            emit(toks[i].text);
            ++i;
            continue;
        }

        // Extend a maximal number run from i. A decimal token joins the run
        // only when flanked by number tokens on both sides.
        int j = i;
        while (j < n) {
            if (IsNumberCat(toks[j].cat)) {
                ++j;
            } else if (toks[j].cat == Cat::kDecimal && j > i &&
                       IsNumberCat(toks[j - 1].cat) && j + 1 < n &&
                       IsNumberCat(toks[j + 1].cat)) {
                ++j;
            } else {
                break;
            }
        }

        // A run led by "năm" + a plain digit is almost always "năm" (year)
        // followed by a year, not a single numeral ("năm hai nghìn không trăm
        // hai mươi" -> "năm 2020"). Drop the leading "năm" and restart.
        if (toks[i].cat == Cat::kUnit && toks[i].text == "năm" &&
            (j - i) >= 3 && i + 1 < n && toks[i + 1].cat == Cat::kUnit) {
            emit(toks[i].text);
            ++i;
            continue;
        }

        int run_len = j - i;
        bool accept = false;
        if (run_len >= 2) {
            accept = true;
        } else if (run_len == 1) {
            const Tok& only = toks[i];
            accept = only.cat == Cat::kTen ||
                     (only.cat == Cat::kUnit && IsSafeSoloUnit(only.text));
        }

        std::string digits = accept ? EvalRun(toks, i, j) : std::string();
        if (digits.empty()) {
            for (int k = i; k < j; ++k) emit(toks[k].text);
            i = j;
            continue;
        }

        // A number run directly followed by "phần trăm" becomes a percentage.
        int next = j;
        if (j + 1 < n && toks[j].text == "phần" && toks[j + 1].text == "trăm") {
            digits.push_back('%');
            next = j + 2;
        }
        emit(digits);
        i = next;
    }
    return out;
}

}  // namespace vietasr
