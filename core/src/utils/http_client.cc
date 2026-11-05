#include "utils/http_client.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <mutex>

#include "vietasr/logger.h"

#if VIETASR_WITH_CURL
#  include <curl/curl.h>
#elif VIETASR_WITH_WINHTTP
#  include <windows.h>
#  include <winhttp.h>
#endif

namespace vietasr {

namespace {

constexpr const char* kUserAgent = "vietasr/0.1";

using ChunkSink = std::function<bool(const char* data, std::size_t size)>;

class TempPath final {
public:
    static std::string For(const std::string& dest) {
        static std::atomic<unsigned long long> counter{0};
        auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
        return dest + ".partial." + std::to_string(static_cast<unsigned long long>(ticks))
            + "." + std::to_string(counter.fetch_add(1));
    }
};

#if VIETASR_WITH_CURL

class CaBundle final {
public:
    static const std::string& Path() {
        static const std::string path = Locate();
        return path;
    }

private:
    static std::string Locate() {
#if defined(__linux__)
        for (const char* env : {"CURL_CA_BUNDLE", "SSL_CERT_FILE"}) {
            const char* value = std::getenv(env);
            if (value != nullptr && *value != '\0') return value;
        }
        for (const char* candidate : {
                 "/etc/ssl/certs/ca-certificates.crt",
                 "/etc/pki/tls/certs/ca-bundle.crt",
                 "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
                 "/etc/ssl/ca-bundle.pem",
                 "/etc/ssl/cert.pem"}) {
            std::ifstream probe(candidate);
            if (probe.good()) return candidate;
        }
#endif
        return std::string();
    }
};

class CurlTransport final {
public:
    static Status Fetch(const std::string& url, const ChunkSink& sink) {
        GlobalInit();
        CURL* curl = curl_easy_init();
        if (!curl) return Status::Error(-4, "curl_easy_init failed");
        ChunkSink callback = sink;
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &CurlTransport::Write);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &callback);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, kUserAgent);
        curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
        curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1024L);
        curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 60L);
        if (!CaBundle::Path().empty()) {
            curl_easy_setopt(curl, CURLOPT_CAINFO, CaBundle::Path().c_str());
        }
        CURLcode rc = curl_easy_perform(curl);
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        curl_easy_cleanup(curl);
        if (rc != CURLE_OK) {
            return Status::Error(-4, std::string(curl_easy_strerror(rc))
                + " (status=" + std::to_string(http_code) + ", url=" + url + ")");
        }
        return Status::Ok();
    }

private:
    static void GlobalInit() {
        static std::once_flag once;
        std::call_once(once, []() { curl_global_init(CURL_GLOBAL_DEFAULT); });
    }

    static std::size_t Write(char* data, std::size_t size, std::size_t nmemb, void* userp) {
        std::size_t total = size * nmemb;
        auto* callback = static_cast<ChunkSink*>(userp);
        return (*callback)(data, total) ? total : 0;
    }
};

using Transport = CurlTransport;

#elif VIETASR_WITH_WINHTTP

class WinHttpHandle final {
public:
    explicit WinHttpHandle(HINTERNET handle) : handle_(handle) {}
    ~WinHttpHandle() { if (handle_) WinHttpCloseHandle(handle_); }
    WinHttpHandle(const WinHttpHandle&) = delete;
    WinHttpHandle& operator=(const WinHttpHandle&) = delete;
    HINTERNET get() const { return handle_; }
    explicit operator bool() const { return handle_ != nullptr; }

private:
    HINTERNET handle_;
};

class WinHttpTransport final {
public:
    static Status Fetch(const std::string& url, const ChunkSink& sink) {
        std::wstring wide = Widen(url);
        URL_COMPONENTS parts{};
        parts.dwStructSize = sizeof(parts);
        parts.dwSchemeLength = static_cast<DWORD>(-1);
        parts.dwHostNameLength = static_cast<DWORD>(-1);
        parts.dwUrlPathLength = static_cast<DWORD>(-1);
        parts.dwExtraInfoLength = static_cast<DWORD>(-1);
        if (!WinHttpCrackUrl(wide.c_str(), 0, 0, &parts)) {
            return Failure("invalid url", url);
        }
        std::wstring host(parts.lpszHostName, parts.dwHostNameLength);
        std::wstring path(parts.lpszUrlPath, parts.dwUrlPathLength);
        if (parts.lpszExtraInfo && parts.dwExtraInfoLength > 0) {
            path.append(parts.lpszExtraInfo, parts.dwExtraInfoLength);
        }
        bool secure = parts.nScheme == INTERNET_SCHEME_HTTPS;

        WinHttpHandle session(WinHttpOpen(Widen(kUserAgent).c_str(), WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
                                          WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0));
        if (!session) return Failure("WinHttpOpen failed", url);
        WinHttpSetTimeouts(session.get(), 30000, 30000, 60000, 60000);
        WinHttpHandle connection(WinHttpConnect(session.get(), host.c_str(), parts.nPort, 0));
        if (!connection) return Failure("WinHttpConnect failed", url);
        WinHttpHandle request(WinHttpOpenRequest(connection.get(), L"GET", path.c_str(), nullptr,
                                                 WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                                                 secure ? WINHTTP_FLAG_SECURE : 0));
        if (!request) return Failure("WinHttpOpenRequest failed", url);
        if (!WinHttpSendRequest(request.get(), WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                                WINHTTP_NO_REQUEST_DATA, 0, 0, 0)
                || !WinHttpReceiveResponse(request.get(), nullptr)) {
            return Failure("request failed", url);
        }
        DWORD status = 0;
        DWORD length = sizeof(status);
        WinHttpQueryHeaders(request.get(), WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                            WINHTTP_HEADER_NAME_BY_INDEX, &status, &length, WINHTTP_NO_HEADER_INDEX);
        if (status < 200 || status >= 300) {
            return Status::Error(-4, "HTTP status " + std::to_string(status) + " (url=" + url + ")");
        }
        std::string buffer;
        for (;;) {
            DWORD available = 0;
            if (!WinHttpQueryDataAvailable(request.get(), &available)) return Failure("read failed", url);
            if (available == 0) break;
            buffer.resize(available);
            DWORD read = 0;
            if (!WinHttpReadData(request.get(), &buffer[0], available, &read)) return Failure("read failed", url);
            if (read == 0) break;
            if (!sink(buffer.data(), read)) return Status::Error(-3, "write failed (url=" + url + ")");
        }
        return Status::Ok();
    }

private:
    static std::wstring Widen(const std::string& text) {
        if (text.empty()) return std::wstring();
        int n = MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
        std::wstring out(static_cast<std::size_t>(n), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), &out[0], n);
        return out;
    }

    static Status Failure(const char* what, const std::string& url) {
        return Status::Error(-4, std::string(what) + " (winhttp error " + std::to_string(GetLastError())
            + ", url=" + url + ")");
    }
};

using Transport = WinHttpTransport;

#endif

}

class HttpClient::Impl {};

HttpClient::HttpClient() : impl_(new Impl()) {}
HttpClient::~HttpClient() { delete impl_; }

Status HttpClient::Get(const std::string& url, std::vector<unsigned char>* body) {
    if (!body) return Status::Error(-1, "HttpClient::Get: null body");
#if VIETASR_WITH_CURL || VIETASR_WITH_WINHTTP
    body->clear();
    Status status = Transport::Fetch(url, [body](const char* data, std::size_t size) {
        const auto* begin = reinterpret_cast<const unsigned char*>(data);
        body->insert(body->end(), begin, begin + size);
        return true;
    });
    if (!status.ok()) return Status::Error(status.code(), "HTTP GET failed: " + status.message());
    return Status::Ok();
#else
    VIETASR_LOG_WARN("http") << "HTTP disabled at build time; url=" << url;
    return Status::Error(-4, "http_client: built without an HTTP backend");
#endif
}

Status HttpClient::Download(const std::string& url, const std::string& dest_path) {
#if VIETASR_WITH_CURL || VIETASR_WITH_WINHTTP
    std::string tmp = TempPath::For(dest_path);
    Status status = Status::Ok();
    {
        std::ofstream stream(tmp, std::ios::binary | std::ios::trunc);
        if (!stream) return Status::Error(-3, "cannot open temp file: " + tmp);
        status = Transport::Fetch(url, [&stream](const char* data, std::size_t size) {
            stream.write(data, static_cast<std::streamsize>(size));
            return static_cast<bool>(stream);
        });
    }
    if (!status.ok()) {
        std::remove(tmp.c_str());
        return Status::Error(status.code(), status.message());
    }
#if defined(_WIN32)
    std::remove(dest_path.c_str());
#endif
    if (std::rename(tmp.c_str(), dest_path.c_str()) != 0) {
        std::remove(tmp.c_str());
        return Status::Error(-3, "rename failed: " + tmp + " -> " + dest_path);
    }
    return Status::Ok();
#else
    VIETASR_LOG_WARN("http") << "HTTP disabled at build time; would download " << url
                              << " -> " << dest_path;
    return Status::Error(-4, "http_client: built without an HTTP backend");
#endif
}

}
