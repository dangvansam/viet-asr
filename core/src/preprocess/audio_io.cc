#include "preprocess/audio_io.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

namespace vietasr {

namespace {

template <typename T>
T ReadLittleEndian(const unsigned char* data) {
    T value = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        value |= static_cast<T>(data[i]) << (8 * i);
    }
    return value;
}

}

Status AudioIo::ReadWav(const std::string& path, AudioClip* out) {
    if (!out) return Status::Error(-1, "AudioIo::ReadWav: null out");
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return Status::Error(-3, "cannot open WAV: " + path);

    unsigned char riff[12];
    stream.read(reinterpret_cast<char*>(riff), 12);
    if (stream.gcount() < 12 ||
        std::memcmp(riff, "RIFF", 4) != 0 ||
        std::memcmp(riff + 8, "WAVE", 4) != 0) {
        return Status::Error(-3, "not a RIFF/WAVE file: " + path);
    }

    bool fmt_parsed = false;
    std::uint16_t format = 0;
    std::uint16_t channels = 0;
    std::uint32_t sample_rate = 0;
    std::uint16_t bits_per_sample = 0;

    while (stream) {
        unsigned char chunk_hdr[8];
        stream.read(reinterpret_cast<char*>(chunk_hdr), 8);
        if (stream.gcount() < 8) {
            return Status::Error(-3, "unexpected EOF while scanning chunks");
        }
        std::uint32_t chunk_size = ReadLittleEndian<std::uint32_t>(chunk_hdr + 4);

        if (std::memcmp(chunk_hdr, "fmt ", 4) == 0) {
            std::vector<unsigned char> fmt(chunk_size);
            stream.read(reinterpret_cast<char*>(fmt.data()), chunk_size);
            if (chunk_size < 16) return Status::Error(-3, "fmt chunk too small");
            format          = ReadLittleEndian<std::uint16_t>(fmt.data() + 0);
            channels        = ReadLittleEndian<std::uint16_t>(fmt.data() + 2);
            sample_rate     = ReadLittleEndian<std::uint32_t>(fmt.data() + 4);
            bits_per_sample = ReadLittleEndian<std::uint16_t>(fmt.data() + 14);
            fmt_parsed = true;
            if (chunk_size % 2 == 1) stream.seekg(1, std::ios::cur);
            continue;
        }

        if (std::memcmp(chunk_hdr, "data", 4) == 0) {
            if (!fmt_parsed) return Status::Error(-3, "data chunk before fmt");
            if (format != 1) return Status::Error(-3, "only PCM WAV supported (format=1)");
            if (bits_per_sample != 16) return Status::Error(-3, "only 16-bit PCM supported");
            if (channels == 0) return Status::Error(-3, "channels=0");

            std::vector<unsigned char> samples(chunk_size);
            stream.read(reinterpret_cast<char*>(samples.data()), chunk_size);

            std::size_t sample_count = chunk_size / 2;
            out->pcm.resize(sample_count / channels);
            out->sample_rate = static_cast<float>(sample_rate);

            std::size_t out_idx = 0;
            for (std::size_t i = 0; i < sample_count; i += channels) {
                std::int32_t mixed = 0;
                for (std::uint16_t ch = 0; ch < channels; ++ch) {
                    std::int16_t sample = static_cast<std::int16_t>(
                        ReadLittleEndian<std::uint16_t>(samples.data() + (i + ch) * 2));
                    mixed += sample;
                }
                mixed /= channels;
                out->pcm[out_idx++] = static_cast<float>(mixed) / 32768.0f;
            }
            return Status::Ok();
        }

        stream.seekg(chunk_size, std::ios::cur);
        if (chunk_size % 2 == 1) stream.seekg(1, std::ios::cur);
    }

    return Status::Error(-3, "no data chunk found");
}

}
