#ifndef VIETASR_UTILS_FS_H
#define VIETASR_UTILS_FS_H

#include <string>
#include <vector>

namespace vietasr {

class FileSystem final {
public:
    static std::string DefaultCacheDir();
    static std::string EnvOrDefault(const char* key, const std::string& fallback);

    static std::string JoinPath(const std::string& a, const std::string& b);
    static bool        EnsureDirectory(const std::string& path);
    static bool        FileExists(const std::string& path);
    static long long   FileSize(const std::string& path);

    static bool ReadFile(const std::string& path, std::vector<unsigned char>* out);
    static bool WriteFile(const std::string& path, const unsigned char* data, std::size_t len);
};

}

#endif
