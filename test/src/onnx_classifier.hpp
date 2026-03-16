#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "classifier_common.hpp"

namespace portable_cls {
namespace detail {

class ScopedStderrSilencer {
public:
    explicit ScopedStderrSilencer(bool enabled = true) : enabled_(enabled) {
        if (!enabled_) return;
        std::fflush(stderr);
#if defined(_WIN32)
        const int stderr_fd = _fileno(stderr);
        if (stderr_fd < 0) return;
        saved_fd_ = _dup(stderr_fd);
        if (saved_fd_ < 0) return;
        null_fd_ = _open("NUL", _O_WRONLY);
        if (null_fd_ < 0) return;
        if (_dup2(null_fd_, stderr_fd) < 0) {
            cleanup_no_restore();
            return;
        }
        active_ = true;
#else
        saved_fd_ = ::dup(STDERR_FILENO);
        if (saved_fd_ < 0) return;
        null_fd_ = ::open("/dev/null", O_WRONLY);
        if (null_fd_ < 0) return;
        if (::dup2(null_fd_, STDERR_FILENO) < 0) {
            cleanup_no_restore();
            return;
        }
        active_ = true;
#endif
    }

    ~ScopedStderrSilencer() {
        restore();
    }

    ScopedStderrSilencer(const ScopedStderrSilencer&) = delete;
    ScopedStderrSilencer& operator=(const ScopedStderrSilencer&) = delete;

private:
    void restore() {
        if (!active_) {
            cleanup_no_restore();
            return;
        }
        std::fflush(stderr);
#if defined(_WIN32)
        const int stderr_fd = _fileno(stderr);
        if (stderr_fd >= 0 && saved_fd_ >= 0) {
            _dup2(saved_fd_, stderr_fd);
        }
        if (saved_fd_ >= 0) _close(saved_fd_);
        if (null_fd_ >= 0) _close(null_fd_);
#else
        if (saved_fd_ >= 0) {
            ::dup2(saved_fd_, STDERR_FILENO);
        }
        if (saved_fd_ >= 0) ::close(saved_fd_);
        if (null_fd_ >= 0) ::close(null_fd_);
#endif
        saved_fd_ = -1;
        null_fd_ = -1;
        active_ = false;
    }

    void cleanup_no_restore() {
#if defined(_WIN32)
        if (saved_fd_ >= 0) _close(saved_fd_);
        if (null_fd_ >= 0) _close(null_fd_);
#else
        if (saved_fd_ >= 0) ::close(saved_fd_);
        if (null_fd_ >= 0) ::close(null_fd_);
#endif
        saved_fd_ = -1;
        null_fd_ = -1;
        active_ = false;
    }

    bool enabled_ = true;
    bool active_ = false;
    int saved_fd_ = -1;
    int null_fd_ = -1;
};

}  // namespace detail

class OnnxClassifier {
public:
    struct Config : public CommonConfig {
        bool enable_cuda = false;              // requires ORT built with CUDA EP
        bool suppress_stderr_on_load = true;   // hide duplicate-schema spam during ORT init/session load
    };

    bool load(const fs::path& model_path,
              const fs::path& labels_path,
              const Config& cfg) {
        cfg_ = cfg;
        labels_ = read_labels(labels_path);

        session_options_ = Ort::SessionOptions();
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options_.SetIntraOpNumThreads(std::max(1, cfg_.num_threads));
        session_options_.SetInterOpNumThreads(1);
#if defined(ORT_API_VERSION) && ORT_API_VERSION >= 8
        session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
#endif

        const std::string model_string = model_path.string();

        {
            detail::ScopedStderrSilencer silencer(cfg_.suppress_stderr_on_load);
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "portable_cls");
#if defined(_WIN32)
            const std::wstring model_w = fs::path(model_path).wstring();
            session_ = std::make_unique<Ort::Session>(*env_, model_w.c_str(), session_options_);
#else
            session_ = std::make_unique<Ort::Session>(*env_, model_string.c_str(), session_options_);
#endif
        }

        Ort::AllocatorWithDefaultOptions allocator;
        input_names_.clear();
        output_names_.clear();
        input_name_storage_.clear();
        output_name_storage_.clear();

        const size_t num_inputs = session_->GetInputCount();
        const size_t num_outputs = session_->GetOutputCount();
        if (num_inputs < 1 || num_outputs < 1) {
            throw std::runtime_error("ONNX model must have at least one input and one output");
        }

        for (size_t i = 0; i < num_inputs; ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_name_storage_.push_back(name.get());
            input_names_.push_back(input_name_storage_.back().c_str());
        }
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_name_storage_.push_back(name.get());
            output_names_.push_back(output_name_storage_.back().c_str());
        }

        const auto input_type_info = session_->GetInputTypeInfo(0);
        const auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        if (shape.size() != 4) {
            throw std::runtime_error(
                "Expected ONNX classifier input shape rank 4, got rank " + std::to_string(shape.size()));
        }

        input_shape_ = std::move(shape);
        if (input_shape_[0] < 1) input_shape_[0] = 1;
        if (input_shape_[1] < 1) input_shape_[1] = 3;
        if (input_shape_[2] < 1) input_shape_[2] = cfg_.input_height;
        if (input_shape_[3] < 1) input_shape_[3] = cfg_.input_width;

        return true;
    }

    bool load(const fs::path& model_path,
              const fs::path& labels_path) {
        return load(model_path, labels_path, Config{});
    }

    Result classify(const cv::Mat& image_bgr, int topk_override = -1) const {
        if (!session_) {
            throw std::runtime_error("ONNX model is not loaded");
        }
        const cv::Mat prepared = preprocess_image(image_bgr, cfg_);
        std::vector<float> input_tensor_values = image_to_nchw(prepared, cfg_);

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            input_shape_.data(),
            input_shape_.size());

        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            &input_tensor,
            1,
            output_names_.data(),
            1);

        if (outputs.empty() || !outputs[0].IsTensor()) {
            throw std::runtime_error("ONNX inference did not return a tensor output");
        }

        const auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
        const size_t count = out_info.GetElementCount();
        const float* out_ptr = outputs[0].GetTensorData<float>();
        std::vector<float> raw(out_ptr, out_ptr + count);
        if (raw.empty()) {
            throw std::runtime_error("ONNX output tensor is empty");
        }

        Result r;
        r.probabilities = probabilities_from_output(raw);
        const int use_topk = (topk_override > 0) ? topk_override : cfg_.topk;
        r.topk = select_topk(r.probabilities, labels_, use_topk);
        if (!r.topk.empty()) {
            r.best_index = r.topk.front().index;
            r.best_probability = r.topk.front().probability;
            r.best_label = r.topk.front().label;
        }
        return r;
    }

    const Config& config() const { return cfg_; }
    const std::vector<std::string>& labels() const { return labels_; }

private:
    static std::vector<float> image_to_nchw(const cv::Mat& prepared_bgr, const CommonConfig& cfg) {
        cv::Mat rgb;
        cv::cvtColor(prepared_bgr, rgb, cv::COLOR_BGR2RGB);

        cv::Mat rgb_float;
        rgb.convertTo(rgb_float, CV_32F);

        const int h = rgb_float.rows;
        const int w = rgb_float.cols;
        std::vector<float> chw(static_cast<size_t>(3) * h * w);

        std::vector<cv::Mat> channels;
        cv::split(rgb_float, channels);
        for (int c = 0; c < 3; ++c) {
            const float mean = cfg.mean_vals[c];
            const float norm = cfg.norm_vals[c];
            const float* src = channels[c].ptr<float>(0);
            float* dst = chw.data() + static_cast<size_t>(c) * h * w;
            const size_t count = static_cast<size_t>(h) * w;
            for (size_t i = 0; i < count; ++i) {
                dst[i] = (src[i] - mean) * norm;
            }
        }
        return chw;
    }

    std::unique_ptr<Ort::Env> env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<int64_t> input_shape_;
    std::vector<std::string> labels_;
    Config cfg_{};

    std::vector<std::string> input_name_storage_;
    std::vector<std::string> output_name_storage_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
};

}  // namespace portable_cls
