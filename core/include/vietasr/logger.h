#ifndef VIETASR_LOGGER_H
#define VIETASR_LOGGER_H

#include <sstream>
#include <string>

namespace vietasr {

enum class LogLevel : int {
    kTrace = 0,
    kDebug = 1,
    kInfo  = 2,
    kWarn  = 3,
    kError = 4,
    kOff   = 5
};

class Logger final {
public:
    static Logger& Instance();

    void SetLevel(LogLevel level);
    LogLevel level() const;

    void Log(LogLevel level, const char* module, const std::string& message);

private:
    Logger();
    ~Logger();
    class Impl;
    Impl* impl_;
};

class LogStream final {
public:
    LogStream(LogLevel level, const char* module)
        : level_(level), module_(module) {}
    ~LogStream() {
        Logger::Instance().Log(level_, module_, stream_.str());
    }

    template <typename T>
    LogStream& operator<<(const T& value) {
        stream_ << value;
        return *this;
    }

private:
    LogLevel level_;
    const char* module_;
    std::ostringstream stream_;
};

#define VIETASR_LOG_TRACE(mod) ::vietasr::LogStream(::vietasr::LogLevel::kTrace, mod)
#define VIETASR_LOG_DEBUG(mod) ::vietasr::LogStream(::vietasr::LogLevel::kDebug, mod)
#define VIETASR_LOG_INFO(mod)  ::vietasr::LogStream(::vietasr::LogLevel::kInfo,  mod)
#define VIETASR_LOG_WARN(mod)  ::vietasr::LogStream(::vietasr::LogLevel::kWarn,  mod)
#define VIETASR_LOG_ERROR(mod) ::vietasr::LogStream(::vietasr::LogLevel::kError, mod)

}

#endif
