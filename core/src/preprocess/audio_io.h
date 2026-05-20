#ifndef VIETASR_PREPROCESS_AUDIO_IO_H
#define VIETASR_PREPROCESS_AUDIO_IO_H

#include <string>

#include "vietasr/types.h"

namespace vietasr {

class AudioIo final {
public:
    static Status ReadWav(const std::string& path, AudioClip* out);
};

}

#endif
