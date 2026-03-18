
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "classifier_common.hpp"

namespace vision_app {

class OnnxClassifier {
public:
    struct Config {
        int input_width = 128;
        int input_height = 128;
        std::string preprocess = "crop";
        int topk = 5;
    };

    bool load(const std::string& model_path,
              const std::string& labels_path,
              const Config& cfg,
              std::string& err) {
        cfg_ = cfg;
        if (!load_labels_txt(labels_path, labels_, err)) return false;
        try {
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "vision_app");
            Ort::SessionOptions so;
            so.SetIntraOpNumThreads(1);
            so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), so);
            Ort::AllocatorWithDefaultOptions allocator;
            auto in = session_->GetInputNameAllocated(0, allocator);
            input_name_ = in.get();
            auto out = session_->GetOutputNameAllocated(0, allocator);
            output_name_ = out.get();
            err.clear();
            return true;
        } catch (const std::exception& e) {
            err = e.what();
            return false;
        }
    }

    ClassifyResult classify(const cv::Mat& image_bgr) const {
        ClassifyResult r;
        if (!session_) { r.summary = "onnx not loaded"; return r; }
        cv::Mat prep = preprocess_bgr(image_bgr, cfg_.input_width, cfg_.input_height, cfg_.preprocess);
        if (prep.empty()) { r.summary = "empty roi"; return r; }

        std::vector<float> nchw(3 * cfg_.input_width * cfg_.input_height);
        for (int y = 0; y < prep.rows; ++y) {
            for (int x = 0; x < prep.cols; ++x) {
                const cv::Vec3b bgr = prep.at<cv::Vec3b>(y, x);
                const size_t idx = static_cast<size_t>(y) * prep.cols + x;
                nchw[idx] = static_cast<float>(bgr[2]) / 255.0f;
                nchw[prep.rows * prep.cols + idx] = static_cast<float>(bgr[1]) / 255.0f;
                nchw[2 * prep.rows * prep.cols + idx] = static_cast<float>(bgr[0]) / 255.0f;
            }
        }

        try {
            const auto t0 = std::chrono::steady_clock::now();
            std::array<int64_t,4> shape{1,3,cfg_.input_height,cfg_.input_width};
            Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input = Ort::Value::CreateTensor<float>(mi, nchw.data(), nchw.size(), shape.data(), shape.size());
            const char* in_names[] = { input_name_.c_str() };
            const char* out_names[] = { output_name_.c_str() };
            auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
            const auto t1 = std::chrono::steady_clock::now();
            r.infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            const auto& out = outputs.front();
            const float* ptr = out.GetTensorData<float>();
            auto shape_info = out.GetTensorTypeAndShapeInfo();
            size_t n = shape_info.GetElementCount();
            std::vector<float> scores(ptr, ptr + n);
            auto idx = topk_indices(scores, cfg_.topk);
            r.ok = true;
            for (int i : idx) {
                ClassHit h;
                h.index = i;
                h.score = scores[i];
                h.label = (i >= 0 && i < static_cast<int>(labels_.size())) ? labels_[i] : ("cls_" + std::to_string(i));
                r.topk.push_back(h);
            }
            if (!r.topk.empty()) r.best = r.topk.front();
            r.summary = make_summary(r);
            return r;
        } catch (const std::exception& e) {
            r.summary = e.what();
            return r;
        }
    }

private:
    Config cfg_;
    std::vector<std::string> labels_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;
};

} // namespace vision_app
