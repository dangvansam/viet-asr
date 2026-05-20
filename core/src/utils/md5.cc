#include "utils/md5.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace vietasr {

namespace {

constexpr std::uint32_t kT[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
};

constexpr std::uint32_t kS[64] = {
    7,12,17,22, 7,12,17,22, 7,12,17,22, 7,12,17,22,
    5, 9,14,20, 5, 9,14,20, 5, 9,14,20, 5, 9,14,20,
    4,11,16,23, 4,11,16,23, 4,11,16,23, 4,11,16,23,
    6,10,15,21, 6,10,15,21, 6,10,15,21, 6,10,15,21
};

inline std::uint32_t LeftRotate(std::uint32_t x, std::uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

}

Md5::Md5() : bit_count_(0), buffered_(0) {
    state_[0] = 0x67452301;
    state_[1] = 0xefcdab89;
    state_[2] = 0x98badcfe;
    state_[3] = 0x10325476;
}

void Md5::Transform(const unsigned char block[64]) {
    std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    std::uint32_t m[16];
    for (int i = 0; i < 16; ++i) {
        m[i] = static_cast<std::uint32_t>(block[i*4])
             | (static_cast<std::uint32_t>(block[i*4 + 1]) << 8)
             | (static_cast<std::uint32_t>(block[i*4 + 2]) << 16)
             | (static_cast<std::uint32_t>(block[i*4 + 3]) << 24);
    }
    for (int i = 0; i < 64; ++i) {
        std::uint32_t f;
        int g;
        if (i < 16) {
            f = (b & c) | (~b & d);
            g = i;
        } else if (i < 32) {
            f = (d & b) | (~d & c);
            g = (5*i + 1) % 16;
        } else if (i < 48) {
            f = b ^ c ^ d;
            g = (3*i + 5) % 16;
        } else {
            f = c ^ (b | ~d);
            g = (7*i) % 16;
        }
        std::uint32_t temp = d;
        d = c;
        c = b;
        b = b + LeftRotate(a + f + kT[i] + m[g], kS[i]);
        a = temp;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
}

void Md5::Update(const unsigned char* data, std::size_t len) {
    bit_count_ += static_cast<std::uint64_t>(len) * 8;
    while (len > 0) {
        std::size_t fill = 64 - buffered_;
        std::size_t copy = len < fill ? len : fill;
        std::memcpy(buffer_ + buffered_, data, copy);
        buffered_ += copy;
        data += copy;
        len  -= copy;
        if (buffered_ == 64) {
            Transform(buffer_);
            buffered_ = 0;
        }
    }
}

void Md5::Final(unsigned char digest[16]) {
    static const unsigned char kPad[64] = {0x80};
    std::uint64_t saved_bits = bit_count_;
    std::size_t pad_len = (buffered_ < 56) ? (56 - buffered_) : (120 - buffered_);
    Update(kPad, pad_len);
    unsigned char length_bytes[8];
    for (int i = 0; i < 8; ++i) {
        length_bytes[i] = static_cast<unsigned char>(saved_bits >> (8*i));
    }
    Update(length_bytes, 8);
    for (int i = 0; i < 4; ++i) {
        digest[i*4]   = static_cast<unsigned char>(state_[i]);
        digest[i*4+1] = static_cast<unsigned char>(state_[i] >> 8);
        digest[i*4+2] = static_cast<unsigned char>(state_[i] >> 16);
        digest[i*4+3] = static_cast<unsigned char>(state_[i] >> 24);
    }
}

std::string Md5::HexDigest() {
    unsigned char digest[16];
    Final(digest);
    char hex[33];
    for (int i = 0; i < 16; ++i) {
        std::snprintf(hex + i*2, 3, "%02x", digest[i]);
    }
    hex[32] = '\0';
    return std::string(hex);
}

std::string Md5::HashBytes(const unsigned char* data, std::size_t len) {
    Md5 hasher;
    hasher.Update(data, len);
    return hasher.HexDigest();
}

std::string Md5::HashFile(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return "";
    Md5 hasher;
    constexpr std::size_t kBufSize = 1 << 16;
    std::vector<unsigned char> buf(kBufSize);
    while (stream) {
        stream.read(reinterpret_cast<char*>(buf.data()), kBufSize);
        std::streamsize got = stream.gcount();
        if (got > 0) hasher.Update(buf.data(), static_cast<std::size_t>(got));
    }
    return hasher.HexDigest();
}

}
