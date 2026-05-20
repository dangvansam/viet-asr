#include "utils/fs.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace vietasr {

namespace fs = std::filesystem;

std::string FileSystem::EnvOrDefault(const char* key, const std::string& fallback) {
    const char* value = std::getenv(key);
    if (value && value[0]) return value;
    return fallback;
}

std::string FileSystem::DefaultCacheDir() {
    std::string override_dir = EnvOrDefault("VIETASR_MODEL_DIR", "");
    if (!override_dir.empty()) return override_dir;

#if defined(_WIN32)
    std::string local = EnvOrDefault("LOCALAPPDATA", "");
    if (!local.empty()) return JoinPath(local, "vietasr");
    return "vietasr_cache";
#elif defined(__APPLE__)
    std::string home = EnvOrDefault("HOME", "");
    if (!home.empty()) return JoinPath(home, "Library/Caches/vietasr");
    return "vietasr_cache";
#else
    std::string xdg = EnvOrDefault("XDG_CACHE_HOME", "");
    if (!xdg.empty()) return JoinPath(xdg, "vietasr");
    std::string home = EnvOrDefault("HOME", "");
    if (!home.empty()) return JoinPath(home, ".cache/vietasr");
    return "vietasr_cache";
#endif
}

std::string FileSystem::JoinPath(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    return (fs::path(a) / b).string();
}

bool FileSystem::EnsureDirectory(const std::string& path) {
    std::error_code ec;
    fs::create_directories(path, ec);
    return !ec;
}

bool FileSystem::FileExists(const std::string& path) {
    std::error_code ec;
    return fs::exists(path, ec) && fs::is_regular_file(path, ec);
}

long long FileSystem::FileSize(const std::string& path) {
    std::error_code ec;
    auto size = fs::file_size(path, ec);
    if (ec) return -1;
    return static_cast<long long>(size);
}

bool FileSystem::ReadFile(const std::string& path, std::vector<unsigned char>* out) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) return false;
    auto end = stream.tellg();
    stream.seekg(0);
    auto size = static_cast<std::size_t>(end);
    out->resize(size);
    if (size > 0) stream.read(reinterpret_cast<char*>(out->data()), size);
    return stream.good() || stream.eof();
}

bool FileSystem::WriteFile(const std::string& path, const unsigned char* data, std::size_t len) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) return false;
    stream.write(reinterpret_cast<const char*>(data), len);
    return stream.good();
}

}
