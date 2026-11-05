# Adding a Module

A module is one folder under [core/src/modules/](../core/src/modules/). To add a new capability (gender classifier, emotion detector, custom ITN model, etc.) you write one folder and add one line of registration. You do not edit any other module, the public C ABI, or any binding.

## Checklist

1. Create folder `core/src/modules/<your_module>/`
2. Subclass `vietasr::Module` in `<your_module>.h` + `<your_module>.cc`
3. If the module needs model files, host them on Hugging Face and pin them in a `models.json` (see [Model files](#model-files))
4. Add a unit test `<your_module>_test.cc` and register it in [core/tests/CMakeLists.txt](../core/tests/CMakeLists.txt)
5. Add a `README.md` describing inputs, outputs, latency, and model size
6. Register with the `VIETASR_REGISTER_MODULE` macro in your .cc file
7. Add an entry in [core/CMakeLists.txt](../core/CMakeLists.txt) under the appropriate option

That is the whole list.

## Skeleton

`core/src/modules/<name>/<name>.h`:

```cpp
#ifndef VIETASR_MODULES_GENDER_GENDER_H
#define VIETASR_MODULES_GENDER_GENDER_H

#include "vietasr/module.h"

namespace vietasr {

class GenderModule final : public Module {
public:
    const char* name() const override { return "gender"; }

    Status Init(const ModuleConfig& config,
                ModelManager* models,
                Engine* engine) override;

    std::unique_ptr<Module> Clone() const override {
        auto copy = std::make_unique<GenderModule>();
        copy->engine_ = engine_;
        copy->model_dir_ = model_dir_;
        return copy;
    }

    void OnSegment(const Segment& segment, ResultBuilder* out) override;
    void OnFinalize(ResultBuilder* out) override;

private:
    Engine* engine_ = nullptr;
    std::string model_dir_;
};

}

#endif
```

`core/src/modules/<name>/<name>.cc`:

```cpp
#include "modules/gender/gender.h"

#include "vietasr/logger.h"

namespace vietasr {

Status GenderModule::Init(const ModuleConfig& config,
                          ModelManager* models,
                          Engine* engine) {
    (void)config;
    engine_ = engine;
    ModelBundle bundle;
    bundle.module = "gender";
    bundle.version = "1";
    bundle.files.push_back({"embedder.onnx",
        "https://huggingface.co/<org>/<repo>/resolve/<revision>/embedder.onnx",
        "abcdef0123456789abcdef0123456789", 8123456});
    return models->EnsureBundle(bundle, &model_dir_);
}

void GenderModule::OnSegment(const Segment& segment, ResultBuilder* out) {
    (void)segment; (void)out;
}

void GenderModule::OnFinalize(ResultBuilder* out) {
    out->SetField("gender",
                  R"({"value":"M","score":0.91})");
}

VIETASR_REGISTER_MODULE("gender", GenderModule)

}
```

`EnsureBundle` downloads every missing file into `<cache_dir>/<module>/<version>/`, verifies the MD5, and returns that folder. A module that cannot work without its model should return the error from `Init`, so building the pipeline fails loudly.

## Model files

Do not commit model weights to the repository. Upload them to Hugging Face and pin them
by revision, the way the `vietasr` module does with
[core/src/modules/vietasr/models.json](../core/src/modules/vietasr/models.json) (repo,
revision, and per-file size, SHA-256 and MD5) and
[model_fetcher.cc](../core/src/modules/vietasr/model_fetcher.cc). Files of a few MB can
instead be embedded at build time, like the VAD models (`core/cmake/EmbedVadModel.cmake`).
Platforms without a system HTTP client (Android, iOS) need the embedded route.

## Hook reference

Override only what you need. Other hooks default to no-ops.

| Hook | When called | Use for |
|---|---|---|
| `OnFrame` | every audio chunk (the whole clip in batch) | audio stats, VAD, ASR |
| `OnSegment` | batch mode: one segment covering the clip | per-utterance classifiers |
| `OnFinalize` | end of stream / file | summary fields, aggregated statistics |
| `OnFeature`, `OnLogits`, `OnText` | declared, **not dispatched yet** | reserved for shared features / encoder output / text post-processing |

`Clone()` must return a fresh instance that shares read-only state (loaded models) but has its own per-stream state: every streaming `Session` clones the pipeline's modules.

## Result schema contract

Write into the `ResultBuilder` with `SetField(dotted_path, json_value)`. Pick a top-level key that is short and unique to your module. Document it in your module's README. Bindings expose the field automatically — no binding code changes needed.

```cpp
out->SetField("emotion",
              R"({"value":"happy","score":0.84})");
```

## Testing

Write a GoogleTest file next to the module and register it in
[core/tests/CMakeLists.txt](../core/tests/CMakeLists.txt) with `vietasr_add_test(...)` +
`gtest_discover_tests(...)`; run everything with `ctest` from the build folder:

```cpp
#include <gtest/gtest.h>
#include "modules/gender/gender.h"

TEST(GenderModule, BasicMaleClip) {
    // load fixture, run module, assert result["gender"]["value"] == "M"
}
```


## Don'ts

- Do not edit other modules' files
- Do not edit `vietasr.h` (the C ABI). New fields go in JSON
- Do not call `printf` / `std::cout`. Use `VIETASR_LOG_*` macros
- Do not duplicate audio preprocessing. Reuse `preprocess/` (`KaldiFbank`, `AudioResampler`, …)
- Do not commit model files. Host them on Hugging Face and pin revision + checksums
- Do not assume a backend. Code against the `Engine` interface
