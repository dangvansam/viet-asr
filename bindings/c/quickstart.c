#include <stdio.h>
#include <vietasr.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <wav-file>\n", argv[0]);
        fprintf(stderr, "presets: %s\n", vietasr_list_presets());
        fprintf(stderr, "modules: %s\n", vietasr_list_modules());
        return 1;
    }

    VietasrPipeline* pipeline = vietasr_pipeline_preset("transcribe");
    if (pipeline == NULL) {
        fprintf(stderr, "preset failed: %s\n", vietasr_last_error());
        return 2;
    }

    const char* result = vietasr_transcribe_file(pipeline, argv[1]);
    if (result == NULL) {
        fprintf(stderr, "transcribe failed: %s\n", vietasr_last_error());
        vietasr_pipeline_free(pipeline);
        return 3;
    }

    printf("%s\n", result);
    vietasr_pipeline_free(pipeline);
    return 0;
}
