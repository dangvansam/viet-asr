# Adding a Module

A module is one folder under [core/src/modules/](../core/src/modules/). To add a new capability (gender classifier, emotion detector, custom ITN model, etc.) you write one folder and add one line of registration. You do not edit any other module, the public C ABI, or any binding.

## Checklist

1. Create folder `core/src/modules/<your_module>/`
2. Subclass `vietasr::Module` in `<your_module>.h` + `<your_module>.cc`
3. Drop `models.json` with the URL + MD5 of every file your module needs
4. Add a unit test `<your_module>_test.cc` with golden fixtures
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
    engine_ = engine;
    ModelBundle bundle;
    auto status = models->LookupBundle("gender", &bundle);
    if (!status.ok()) return status;
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

`core/src/modules/<name>/models.json`:

```json
{
  "module": "gender",
  "version": "1",
  "files": [
    {
      "name": "embedder.onnx",
      "url":  "https://cdn.vietasr.io/gender/1/embedder.onnx",
      "md5":  "abcdef0123456789abcdef0123456789",
      "size_bytes": 8123456
    }
  ]
}
```

## Hook reference

Override only what you need. Other hooks default to no-ops.

| Hook | When called | Use for |
|---|---|---|
| `OnFrame` | every audio chunk | low-level audio stats (noise dB, energy) |
| `OnFeature` | every fbank frame | features-based classifiers (gender, dialect) |
| `OnLogits` | every encoder output frame | ASR-adjacent modules |
| `OnSegment` | endpoint or VAD boundary | per-utterance classifiers, diarization |
| `OnText` | text emitted by ASR | punctuation, ITN, language tagging |
| `OnFinalize` | end of stream / file | summary fields, aggregated statistics |

## Result schema contract

Write into the `ResultBuilder` with `SetField(dotted_path, json_value)`. Pick a top-level key that is short and unique to your module. Document it in your module's README. Bindings expose the field automatically — no binding code changes needed.

```cpp
out->SetField("emotion",
              R"({"value":"happy","score":0.84})");
```

## Testing

Drop a fixture WAV in `core/tests/fixtures/` and write a golden test:

```cpp
#include <gtest/gtest.h>
#include "modules/gender/gender.h"

TEST(GenderModule, BasicMaleClip) {
    // load fixture, run module, assert result["gender"]["value"] == "M"
}
```

CI runs all module tests on every PR. A failing module test does not block other modules from shipping.

## Don'ts

- Do not edit other modules' files
- Do not edit `vietasr.h` (the C ABI). New fields go in JSON
- Do not call `printf` / `std::cout`. Use `VIETASR_LOG_*` macros
- Do not duplicate audio preprocessing. Use the shared `OnFeature` / `OnLogits` hooks
- Do not commit model files. Host them and reference by URL + MD5
- Do not assume a backend. Code against the `Engine` interface
