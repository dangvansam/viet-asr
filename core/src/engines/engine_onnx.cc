#include "vietasr/engine.h"

#include "vietasr/logger.h"

#if VIETASR_HAS_ONNXRUNTIME
#  include <onnxruntime_cxx_api.h>
#  include <algorithm>
#  include <memory>
#  include <string>
#  include <unordered_map>
#  include <vector>
#  include "vietasr.h"
#  ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#      define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#      define NOMINMAX
#    endif
#    include <windows.h>
#  endif
// Provider-factory headers ship only inside the matching ONNX Runtime binary.
// core/CMakeLists.txt defines VIETASR_EP_*_AVAILABLE after inspecting the
// fetched runtime, so the same source compiles against any ORT package.
#  ifdef VIETASR_EP_COREML_AVAILABLE
#    include <coreml_provider_factory.h>
#  endif
#  ifdef VIETASR_EP_NNAPI_AVAILABLE
#    include <nnapi_provider_factory.h>
#  endif
#  ifdef VIETASR_EP_DIRECTML_AVAILABLE
#    include <dml_provider_factory.h>
#  endif

namespace vietasr {

namespace {

#ifdef _WIN32
// onnxruntime's file-path Session constructor takes ORTCHAR_T* — wchar_t on
// Windows. Convert the UTF-8 model path to UTF-16 so Unicode paths work too.
std::wstring Utf8ToWide(const std::string& s) {
    if (s.empty()) return std::wstring();
    int n = ::MultiByteToWideChar(CP_UTF8, 0, s.data(),
                                  static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<std::size_t>(n), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()),
                          w.data(), n);
    return w;
}
#endif

class OnnxEngine final : public Engine {
public:
    explicit OnnxEngine(const EpPreference& ep)
        : env_(ORT_LOGGING_LEVEL_WARNING, "vietasr"), ep_pref_(ep) {
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
        ConfigureProviders();
    }

    const char* backend_name() const override { return "onnx"; }

    void SetIntraOpThreads(int threads) override {
        if (threads < 1) threads = 1;
        session_options_.SetIntraOpNumThreads(threads);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (threads == 1) {
            session_options_.AddConfigEntry("session.intra_op.allow_spinning", "0");
        }
    }

    const char* active_provider() const override { return active_provider_.c_str(); }

    Status LoadModel(const std::string& model_path) override {
        try {
#ifdef _WIN32
            std::wstring wpath = Utf8ToWide(model_path);
            session_ = std::make_unique<Ort::Session>(
                env_, wpath.c_str(), session_options_);
#else
            session_ = std::make_unique<Ort::Session>(
                env_, model_path.c_str(), session_options_);
#endif
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
            ONNXTensorElementDataType out_type = info.GetElementType();
            if (out_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                const std::int64_t* src = v.GetTensorData<std::int64_t>();
                for (std::size_t k = 0; k < n_elements; ++k) {
                    (*outputs)[i].data[k] = static_cast<float>(src[k]);
                }
            } else if (out_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
                const std::int32_t* src = v.GetTensorData<std::int32_t>();
                for (std::size_t k = 0; k < n_elements; ++k) {
                    (*outputs)[i].data[k] = static_cast<float>(src[k]);
                }
            } else {
                const float* src = v.GetTensorData<float>();
                std::copy(src, src + n_elements, (*outputs)[i].data.begin());
            }
        }
        return Status::Ok();
    }

    void Reset() override {}

private:
    // Builds the execution-provider chain on session_options_ from ep_pref_.
    // Never throws: a provider that fails to register is logged and skipped,
    // leaving ONNX Runtime's CPU provider as the implicit fallback.
    void ConfigureProviders() {
        std::vector<std::string> available;
        try {
            available = Ort::GetAvailableProviders();
        } catch (const Ort::Exception&) {
            // Treat as CPU-only.
        }

        std::vector<int> wanted;
        if (ep_pref_.ep_enum == VIETASR_EP_AUTO) {
            // Platform priority order; CPU is always the implicit fallback.
#if defined(__APPLE__)
            wanted = {VIETASR_EP_COREML};
#elif defined(__ANDROID__)
            wanted = {VIETASR_EP_NNAPI, VIETASR_EP_XNNPACK};
#elif defined(_WIN32)
#  if defined(_M_ARM64)
            wanted = {VIETASR_EP_QNN};
#  else
            wanted = {VIETASR_EP_CUDA, VIETASR_EP_DIRECTML};
#  endif
#elif defined(__linux__)
            wanted = {VIETASR_EP_CUDA, VIETASR_EP_TENSORRT};
#endif
        } else if (ep_pref_.ep_enum != VIETASR_EP_CPU) {
            wanted = {ep_pref_.ep_enum};
        }

        for (int ep : wanted) {
            if (TryAppendProvider(ep, available)) {
                VIETASR_LOG_INFO("engine_onnx")
                    << "execution provider: " << active_provider_;
                return;
            }
        }

        if (ep_pref_.ep_enum != VIETASR_EP_AUTO
                && ep_pref_.ep_enum != VIETASR_EP_CPU) {
            VIETASR_LOG_INFO("engine_onnx")
                << "requested execution provider '"
                << EpEnumName(ep_pref_.ep_enum)
                << "' unavailable in this build; using CPU";
        }
        // active_provider_ stays "CPUExecutionProvider".
    }

    // Registers one execution provider on session_options_. Returns false if the
    // provider is not compiled into this ORT binary, not present at runtime, or
    // throws while registering.
    bool TryAppendProvider(int ep, const std::vector<std::string>& available) {
        [[maybe_unused]] auto has = [&](const char* name) {
            return std::find(available.begin(), available.end(), name)
                   != available.end();
        };
        try {
            switch (ep) {
#ifdef VIETASR_EP_CUDA_AVAILABLE
            case VIETASR_EP_CUDA: {
                if (!has("CUDAExecutionProvider")) return false;
                OrtCUDAProviderOptions opts{};
                opts.device_id = ep_pref_.device_id;
                session_options_.AppendExecutionProvider_CUDA(opts);
                active_provider_ = "CUDAExecutionProvider";
                return true;
            }
#endif
#ifdef VIETASR_EP_TENSORRT_AVAILABLE
            case VIETASR_EP_TENSORRT: {
                if (!has("TensorrtExecutionProvider")) return false;
                OrtTensorRTProviderOptions opts{};
                opts.device_id = ep_pref_.device_id;
                session_options_.AppendExecutionProvider_TensorRT(opts);
                active_provider_ = "TensorrtExecutionProvider";
                return true;
            }
#endif
#ifdef VIETASR_EP_COREML_AVAILABLE
            case VIETASR_EP_COREML: {
                if (!has("CoreMLExecutionProvider")) return false;
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(
                    session_options_, /*coreml_flags=*/0));
                active_provider_ = "CoreMLExecutionProvider";
                return true;
            }
#endif
#ifdef VIETASR_EP_NNAPI_AVAILABLE
            case VIETASR_EP_NNAPI: {
                if (!has("NnapiExecutionProvider")) return false;
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(
                    session_options_, /*nnapi_flags=*/0));
                active_provider_ = "NnapiExecutionProvider";
                return true;
            }
#endif
#ifdef VIETASR_EP_XNNPACK_AVAILABLE
            case VIETASR_EP_XNNPACK: {
                if (!has("XnnpackExecutionProvider")) return false;
                session_options_.AppendExecutionProvider("XNNPACK", {});
                active_provider_ = "XnnpackExecutionProvider";
                return true;
            }
#endif
#ifdef VIETASR_EP_DIRECTML_AVAILABLE
            case VIETASR_EP_DIRECTML: {
                if (!has("DmlExecutionProvider")) return false;
                // DirectML requires sequential execution with mem pattern off.
                session_options_.DisableMemPattern();
                session_options_.SetExecutionMode(ORT_SEQUENTIAL);
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(
                    session_options_, ep_pref_.device_id));
                active_provider_ = "DmlExecutionProvider";
                return true;
            }
#endif
#ifdef VIETASR_EP_QNN_AVAILABLE
            case VIETASR_EP_QNN: {
                if (!has("QNNExecutionProvider")) return false;
                std::unordered_map<std::string, std::string> opts;
                opts["backend_path"] = "libQnnHtp.so";
                session_options_.AppendExecutionProvider("QNN", opts);
                active_provider_ = "QNNExecutionProvider";
                return true;
            }
#endif
            default:
                return false;
            }
        } catch (const Ort::Exception& exc) {
            VIETASR_LOG_WARN("engine_onnx")
                << "could not enable execution provider '" << EpEnumName(ep)
                << "': " << exc.what() << " — falling back";
            return false;
        }
    }

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
    EpPreference ep_pref_;
    std::string active_provider_{"CPUExecutionProvider"};

    std::vector<std::string> input_names_storage_;
    std::vector<std::string> output_names_storage_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
};

}

std::unique_ptr<Engine> CreateOnnxEngine(const EpPreference& ep) {
    return std::make_unique<OnnxEngine>(ep);
}

std::vector<std::string> OnnxAvailableProviders() {
    try {
        return Ort::GetAvailableProviders();
    } catch (const Ort::Exception&) {
        return {};
    }
}

}

#else

namespace vietasr {

std::unique_ptr<Engine> CreateOnnxEngine(const EpPreference& ep) {
    (void)ep;
    VIETASR_LOG_DEBUG("engine_onnx")
        << "ONNX Runtime backend not compiled in (set -DVIETASR_BACKEND_ONNX=ON and ensure onnxruntime is available)";
    return nullptr;
}

std::vector<std::string> OnnxAvailableProviders() { return {}; }

}

#endif
