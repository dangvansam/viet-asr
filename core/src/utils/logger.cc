#include "vietasr/logger.h"

#include <atomic>
#include <cstdio>
#include <mutex>

namespace vietasr {

class Logger::Impl {
public:
    std::atomic<int> level{static_cast<int>(LogLevel::kInfo)};
    std::mutex sink_mutex;
};

Logger::Logger() : impl_(new Impl()) {}
Logger::~Logger() { delete impl_; }

Logger& Logger::Instance() {
    static Logger instance;
    return instance;
}

void Logger::SetLevel(LogLevel level) {
    Instance().impl_->level.store(static_cast<int>(level));
}

LogLevel Logger::level() const {
    return static_cast<LogLevel>(impl_->level.load());
}

void Logger::Log(LogLevel level, const char* module, const std::string& message) {
    if (static_cast<int>(level) < impl_->level.load()) return;
    static const char* names[] = {"TRACE", "DEBUG", "INFO", "WARN", "ERROR"};
    int idx = static_cast<int>(level);
    if (idx < 0 || idx > 4) return;
    std::lock_guard<std::mutex> lock(impl_->sink_mutex);
    std::fprintf(stderr, "[vietasr][%s][%s] %s\n",
                 names[idx],
                 module ? module : "core",
                 message.c_str());
}

}
