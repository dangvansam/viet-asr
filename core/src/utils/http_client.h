#ifndef VIETASR_UTILS_HTTP_CLIENT_H
#define VIETASR_UTILS_HTTP_CLIENT_H

#include <string>
#include <vector>

#include "vietasr/types.h"

namespace vietasr {

class HttpClient final {
public:
    HttpClient();
    ~HttpClient();

    Status Get(const std::string& url, std::vector<unsigned char>* body);
    Status Download(const std::string& url, const std::string& dest_path);

private:
    class Impl;
    Impl* impl_;
};

}

#endif
