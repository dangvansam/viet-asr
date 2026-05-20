#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vietasr.h>

static int16_t* read_wav_mono_i16(const char* path, int* out_len, int* out_rate) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    unsigned char header[12];
    if (fread(header, 1, 12, f) != 12 ||
        memcmp(header, "RIFF", 4) != 0 ||
        memcmp(header + 8, "WAVE", 4) != 0) {
        fclose(f);
        return NULL;
    }

    int sample_rate = 16000;
    int channels = 1;
    int bits = 16;

    while (1) {
        unsigned char chunk[8];
        if (fread(chunk, 1, 8, f) != 8) {
            fclose(f);
            return NULL;
        }
        uint32_t size = chunk[4] | (chunk[5] << 8) | (chunk[6] << 16) | (chunk[7] << 24);
        if (memcmp(chunk, "fmt ", 4) == 0) {
            unsigned char fmt[64];
            uint32_t want = size < sizeof(fmt) ? size : sizeof(fmt);
            if (fread(fmt, 1, want, f) != want) {
                fclose(f);
                return NULL;
            }
            channels = fmt[2] | (fmt[3] << 8);
            sample_rate = fmt[4] | (fmt[5] << 8) | (fmt[6] << 16) | (fmt[7] << 24);
            bits = fmt[14] | (fmt[15] << 8);
            if (size > want) {
                fseek(f, size - want, SEEK_CUR);
            }
        } else if (memcmp(chunk, "data", 4) == 0) {
            if (bits != 16) {
                fclose(f);
                return NULL;
            }
            int total = (int)(size / 2);
            int16_t* all = malloc((size_t)total * sizeof(int16_t));
            if (fread(all, sizeof(int16_t), (size_t)total, f) != (size_t)total) {
                free(all);
                fclose(f);
                return NULL;
            }
            fclose(f);
            if (channels == 1) {
                *out_len = total;
                *out_rate = sample_rate;
                return all;
            }
            int mono_len = total / channels;
            int16_t* mono = malloc((size_t)mono_len * sizeof(int16_t));
            for (int i = 0; i < mono_len; ++i) {
                int32_t mixed = 0;
                for (int c = 0; c < channels; ++c) {
                    mixed += all[i * channels + c];
                }
                mono[i] = (int16_t)(mixed / channels);
            }
            free(all);
            *out_len = mono_len;
            *out_rate = sample_rate;
            return mono;
        } else {
            fseek(f, size, SEEK_CUR);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <wav-file>\n", argv[0]);
        return 1;
    }

    int len = 0;
    int rate = 0;
    int16_t* pcm = read_wav_mono_i16(argv[1], &len, &rate);
    if (!pcm) {
        fprintf(stderr, "failed to read WAV: %s\n", argv[1]);
        return 2;
    }
    printf("audio: %.2fs @ %d Hz\n", (double)len / rate, rate);

    VietasrPipeline* pipeline = vietasr_pipeline_preset("transcribe");
    if (!pipeline) {
        fprintf(stderr, "preset failed: %s\n", vietasr_last_error());
        free(pcm);
        return 3;
    }

    VietasrSession* session = vietasr_session_new(pipeline, (float)rate);
    int chunk = rate / 1000 * 320;
    for (int offset = 0; offset < len; offset += chunk) {
        int n = (offset + chunk <= len) ? chunk : (len - offset);
        vietasr_accept_waveform_s16(session, pcm + offset, n);
    }

    const char* final_json = vietasr_final_result(session);
    printf("FINAL: %s\n", final_json);

    vietasr_session_free(session);
    vietasr_pipeline_free(pipeline);
    free(pcm);
    return 0;
}
