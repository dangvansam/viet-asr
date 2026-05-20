#ifndef VIETASR_UTILS_MD5_H
#define VIETASR_UTILS_MD5_H

#include <cstdint>
#include <string>

namespace vietasr {

class Md5 final {
public:
    Md5();
    void Update(const unsigned char* data, std::size_t len);
    std::string HexDigest();

    static std::string HashFile(const std::string& path);
    static std::string HashBytes(const unsigned char* data, std::size_t len);

private:
    std::uint32_t state_[4];
    std::uint64_t bit_count_;
    unsigned char buffer_[64];
    std::size_t buffered_;

    void Transform(const unsigned char block[64]);
    void Final(unsigned char digest[16]);
};

}

#endif
