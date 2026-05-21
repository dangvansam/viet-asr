#ifndef VIETASR_RESULT_H
#define VIETASR_RESULT_H

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "vietasr/types.h"

namespace vietasr {

class ResultBuilder final {
public:
    ResultBuilder();
    ~ResultBuilder();

    ResultBuilder(const ResultBuilder&) = delete;
    ResultBuilder& operator=(const ResultBuilder&) = delete;

    void Reset();

    void SetText(const std::string& module, std::string text);
    void SetPartial(std::string text);
    void SetField(const std::string& dotted_path, std::string json_value);
    void AppendSegment(TextSegment segment);
    void AppendSpeaker(std::string speaker_id, double total_time_s);

    void MarkFinal(bool is_final);

    std::string SnapshotJson() const;
    std::string FinalJson();

    // The current canonical transcript text (unescaped, raw UTF-8). Empty when
    // no module has emitted text yet. Used by text-consuming modules such as
    // itn, which read the ASR output and rewrite it in OnFinalize.
    std::string CurrentText() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}

#endif
