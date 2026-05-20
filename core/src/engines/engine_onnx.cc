#include "vietasr/engine.h"

#include "vietasr/logger.h"

#if VIETASR_HAS_ONNXRUNTIME
#  include <onnxruntime_cxx_api.h>
#  include <memory>
#  include <vector>

namespace vietasr {

namespace {

class OnnxEngine final : public Engine {
public:
    OnnxEngine()
        : env_(ORT_LOGGING_LEVEL_WARNING, "vietasr") {
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    }

    const char* backend_name() const override { return "onnx"; }

    Status LoadModel(const std::string& model_path) override {
        try {
            session_ = std::make_unique<Ort::Session>(
                env_, model_path.c_str(), session_options_);
        } catch (const Ort::Exception& exc) {
            return Status::Error(-6,
                std::string("Ort::Session ctor failed: ") + exc.what());
        }
        Status meta = InitSessionMeta();
        if (!meta.ok()) return meta;
        VIETASR_LOG_INFO("engine_onnx")
            << "loaded " << model_path
            << " inputs=" << input_names_.size()
            << " outputs=" << output_names_.size();
        return Status::Ok();
    }

    Status LoadModelFromBuffer(const void* data, std::size_t size) override {
        if (!data || size == 0) {
            return Status::Error(-6, "engine_onnx: empty model buffer");
        }
        try {
            session_ = std::make_unique<Ort::Session>(
                env_, data, size, session_options_);
        } catch (const Ort::Exception& exc) {
            return Status::Error(-6,
                std::string("Ort::Session (buffer) ctor failed: ") + exc.what());
        }
        Status meta = InitSessionMeta();
        if (!meta.ok()) return meta;
        VIETASR_LOG_INFO("engine_onnx")
            << "loaded model from buffer (" << size << " bytes)"
            << " inputs=" << input_names_.size()
            << " outputs=" << output_names_.size();
        return Status::Ok();
    }

    Status Run(const std::vector<Tensor>& inputs,
               std::vector<Tensor>* outputs) override {
        if (!session_) return Status::Error(-6, "engine_onnx: model not loaded");
        if (inputs.size() != input_names_.size()) {
            return Status::Error(-1,
                "engine_onnx: input count mismatch (got "
                + std::to_string(inputs.size()) + ", model expects "
                + std::to_string(input_names_.size()) + ")");
        }

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtDeviceAllocator, OrtMemTypeDefault);

        std::vector<Ort::Value> input_tensors;
        input_tensors.reserve(inputs.size());
        std::vector<std::vector<std::int64_t>> int64_scratch(inputs.size());

        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const Tensor& t = inputs[i];
            try {
                auto type_info = session_->GetInputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                ONNXTensorElementDataType dtype = tensor_info.GetElementType();
                if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
                        memory_info,
                        const_cast<float*>(t.data.data()),
                        t.data.size(),
                        t.shape.data(),
                        t.shape.size()));
                } else if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                    auto& buf = int64_scratch[i];
                    buf.assign(t.data.begin(), t.data.end());
                    input_tensors.emplace_back(Ort::Value::CreateTensor<std::int64_t>(
                        memory_info,
                        buf.data(),
                        buf.size(),
                        t.shape.data(),
                        t.shape.size()));
                } else {
                    return Status::Error(-6,
                        std::string("engine_onnx: input ") + input_names_[i]
                        + " has unsupported dtype id="
                        + std::to_string(static_cast<int>(dtype)));
                }
            } catch (const Ort::Exception& exc) {
                return Status::Error(-6,
                    std::string("engine_onnx: input tensor build failed: ") + exc.what());
            }
        }

        std::vector<Ort::Value> output_tensors;
        try {
            output_tensors = session_->Run(
                Ort::RunOptions{nullptr},
                input_names_.data(),
                input_tensors.data(),
                input_tensors.size(),
                output_names_.data(),
                output_names_.size());
        } catch (const Ort::Exception& exc) {
            return Status::Error(-6,
                std::string("engine_onnx: Run failed: ") + exc.what());
        }

        outputs->clear();
        outputs->resize(output_tensors.size());
        for (std::size_t i = 0; i < output_tensors.size(); ++i) {
            const Ort::Value& v = output_tensors[i];
            auto info = v.GetTensorTypeAndShapeInfo();
            auto shape = info.GetShape();
            std::size_t n_elements = 1;
            for (auto d : shape) n_elements *= static_cast<std::size_t>(d);
            (*outputs)[i].shape = shape;
            (*outputs)[i].data.resize(n_elements);
            const float* src = v.GetTensorData<float>();
            std::copy(src, src + n_elements, (*outputs)[i].data.begin());
        }
        return Status::Ok();
    }

    void Reset() override {}

private:
    Status InitSessionMeta() {
        try {
            Ort::AllocatorWithDefaultOptions allocator;
            input_names_storage_.clear();
            input_names_.clear();
            output_names_storage_.clear();
            output_names_.clear();

            std::size_t n_inputs = session_->GetInputCount();
            std::size_t n_outputs = session_->GetOutputCount();
            input_names_storage_.reserve(n_inputs);
            output_names_storage_.reserve(n_outputs);

            for (std::size_t i = 0; i < n_inputs; ++i) {
                auto name = session_->GetInputNameAllocated(i, allocator);
                input_names_storage_.emplace_back(name.get());
            }
            for (std::size_t i = 0; i < n_outputs; ++i) {
                auto name = session_->GetOutputNameAllocated(i, allocator);
                output_names_storage_.emplace_back(name.get());
            }
            for (const auto& s : input_names_storage_) input_names_.push_back(s.c_str());
            for (const auto& s : output_names_storage_) output_names_.push_back(s.c_str());
        } catch (const Ort::Exception& exc) {
            return Status::Error(-6,
                std::string("engine_onnx: session metadata init failed: ") + exc.what());
        }
        return Status::Ok();
    }

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;

    std::vector<std::string> input_names_storage_;
    std::vector<std::string> output_names_storage_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
};

}

std::unique_ptr<Engine> CreateOnnxEngine() {
    return std::make_unique<OnnxEngine>();
}

}

#else

namespace vietasr {

std::unique_ptr<Engine> CreateOnnxEngine() {
    VIETASR_LOG_DEBUG("engine_onnx")
        << "ONNX Runtime backend not compiled in (set -DVIETASR_BACKEND_ONNX=ON and ensure onnxruntime is available)";
    return nullptr;
}

}

#endif
