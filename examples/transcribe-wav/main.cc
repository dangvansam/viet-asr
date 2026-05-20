#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "vietasr.h"

int main(int argc, char** argv) {
    const char* preset = nullptr;
    const char* wav_path = nullptr;
    std::string modules;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--preset" && i + 1 < argc) {
            preset = argv[++i];
        } else if (arg == "--module" && i + 1 < argc) {
            if (!modules.empty()) modules.push_back(',');
            modules += argv[++i];
        } else if (arg == "--log" && i + 1 < argc) {
            const char* level = argv[++i];
            if (std::strcmp(level, "trace") == 0) vietasr_set_log_level(VIETASR_LOG_TRACE);
            else if (std::strcmp(level, "debug") == 0) vietasr_set_log_level(VIETASR_LOG_DEBUG);
            else if (std::strcmp(level, "info")  == 0) vietasr_set_log_level(VIETASR_LOG_INFO);
            else if (std::strcmp(level, "warn")  == 0) vietasr_set_log_level(VIETASR_LOG_WARN);
            else if (std::strcmp(level, "error") == 0) vietasr_set_log_level(VIETASR_LOG_ERROR);
        } else if (arg == "-h" || arg == "--help") {
            std::string presets = vietasr_list_presets();
            std::string modules = vietasr_list_modules();
            std::fprintf(stderr,
                "Usage: %s [--preset NAME] [--module NAME]... [--log LEVEL] WAV_PATH\n"
                "\n"
                "Examples:\n"
                "  %s sample.wav\n"
                "  %s --preset analytics call.wav\n"
                "  %s --module vad --module vietasr audio.wav\n"
                "\n"
                "Available presets:  %s\n"
                "Available modules:  %s\n",
                argv[0], argv[0], argv[0], argv[0],
                presets.c_str(), modules.c_str());
            return 0;
        } else if (arg.size() > 0 && arg[0] != '-') {
            wav_path = argv[i];
        }
    }

    if (!wav_path) {
        std::string presets = vietasr_list_presets();
        std::string modules = vietasr_list_modules();
        std::fprintf(stderr,
            "Usage: %s [--preset NAME] [--module NAME]... [--log LEVEL] WAV_PATH\n"
            "\n"
            "Available presets:  %s\n"
            "Available modules:  %s\n",
            argv[0], presets.c_str(), modules.c_str());
        return 1;
    }

    VietasrPipeline* pipeline = nullptr;
    if (!modules.empty()) {
        pipeline = vietasr_pipeline_new();
        std::string current;
        for (char ch : modules) {
            if (ch == ',') {
                if (!current.empty()) {
                    if (vietasr_pipeline_add_module(pipeline, current.c_str(), "{}") != VIETASR_OK) {
                        std::fprintf(stderr, "add_module failed: %s — %s\n",
                                     current.c_str(), vietasr_last_error());
                        vietasr_pipeline_free(pipeline);
                        return 2;
                    }
                    current.clear();
                }
            } else {
                current.push_back(ch);
            }
        }
        if (!current.empty()) {
            vietasr_pipeline_add_module(pipeline, current.c_str(), "{}");
        }
        if (vietasr_pipeline_build(pipeline) != VIETASR_OK) {
            std::fprintf(stderr, "build failed: %s\n", vietasr_last_error());
            vietasr_pipeline_free(pipeline);
            return 3;
        }
    } else {
        pipeline = vietasr_pipeline_preset(preset ? preset : "transcribe");
        if (!pipeline) {
            std::fprintf(stderr, "preset failed: %s — %s\n",
                         preset ? preset : "transcribe",
                         vietasr_last_error());
            return 4;
        }
    }

    const char* result = vietasr_transcribe_file(pipeline, wav_path);
    if (result) {
        std::fputs(result, stdout);
        std::fputc('\n', stdout);
    } else {
        std::fprintf(stderr, "transcribe failed: %s\n", vietasr_last_error());
    }

    vietasr_pipeline_free(pipeline);
    return result ? 0 : 5;
}
