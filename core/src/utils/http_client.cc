#include "utils/http_client.h"

#include <cstdio>
#include <fstream>
#include <mutex>

#include "vietasr/logger.h"

#if VIETASR_WITH_CURL
#  include <curl/curl.h>
#endif

namespace vietasr {

namespace {

#if VIETASR_WITH_CURL
std::once_flag g_curl_init_once;

void GlobalCurlInit() {
    std::call_once(g_curl_init_once, []() {
        curl_global_init(CURL_GLOBAL_DEFAULT);
    });
}
#endif

}

class HttpClient::Impl {
public:
    Impl() {
#if VIETASR_WITH_CURL
        GlobalCurlInit();
#endif
    }
};

namespace {

#if VIETASR_WITH_CURL
std::size_t WriteToVector(char* contents, std::size_t size, std::size_t nmemb, void* userp) {
    std::size_t total = size * nmemb;
    auto* body = static_cast<std::vector<unsigned char>*>(userp);
    auto* begin = reinterpret_cast<unsigned char*>(contents);
    body->insert(body->end(), begin, begin + total);
    return total;
}

std::size_t WriteToFile(char* contents, std::size_t size, std::size_t nmemb, void* userp) {
    std::size_t total = size * nmemb;
    auto* stream = static_cast<std::ofstream*>(userp);
    stream->write(contents, static_cast<std::streamsize>(total));
    return total;
}
#endif

}

HttpClient::HttpClient() : impl_(new Impl()) {}
HttpClient::~HttpClient() { delete impl_; }

Status HttpClient::Get(const std::string& url, std::vector<unsigned char>* body) {
#if VIETASR_WITH_CURL
    if (!body) return Status::Error(-1, "HttpClient::Get: null body");
    CURL* curl = curl_easy_init();
    if (!curl) return Status::Error(-4, "curl_easy_init failed");
    body->clear();
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteToVector);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, body);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "vietasr/0.1");
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    CURLcode rc = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);
    if (rc != CURLE_OK) {
        return Status::Error(-4, std::string("HTTP GET failed: ") + curl_easy_strerror(rc)
            + " (status=" + std::to_string(http_code) + ", url=" + url + ")");
    }
    return Status::Ok();
#else
    (void)body;
    VIETASR_LOG_WARN("http") << "HTTP disabled at build time; url=" << url;
    return Status::Error(-4, "http_client: built without libcurl");
#endif
}

Status HttpClient::Download(const std::string& url, const std::string& dest_path) {
#if VIETASR_WITH_CURL
    std::string tmp = dest_path + ".partial";
    {
        std::ofstream stream(tmp, std::ios::binary | std::ios::trunc);
        if (!stream) return Status::Error(-3, "cannot open temp file: " + tmp);
        CURL* curl = curl_easy_init();
        if (!curl) return Status::Error(-4, "curl_easy_init failed");
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteToFile);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &stream);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "vietasr/0.1");
        curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        CURLcode rc = curl_easy_perform(curl);
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        curl_easy_cleanup(curl);
        if (rc != CURLE_OK) {
            std::remove(tmp.c_str());
            return Status::Error(-4, std::string("download failed: ") + curl_easy_strerror(rc)
                + " (status=" + std::to_string(http_code) + ", url=" + url + ")");
        }
    }
    if (std::rename(tmp.c_str(), dest_path.c_str()) != 0) {
        std::remove(tmp.c_str());
        return Status::Error(-3, "rename failed: " + tmp + " -> " + dest_path);
    }
    return Status::Ok();
#else
    VIETASR_LOG_WARN("http") << "HTTP disabled at build time; would download " << url
                              << " -> " << dest_path;
    return Status::Error(-4, "http_client: built without libcurl");
#endif
}

}
