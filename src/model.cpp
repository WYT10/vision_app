#include "model.hpp"

#include <chrono>
#include <memory>
#include <stdexcept>

#include "deploy.hpp"
#include "classifier_common.hpp"
#if defined(VISION_APP_HAS_ONNX)
#include "onnx_classifier.hpp"
#endif
#if defined(VISION_APP_HAS_NCNN)
#include "ncnn_classifier.hpp"
#endif

namespace vision_app {
namespace {

class RuntimeIface {
public:
    virtual ~RuntimeIface() = default;
    virtual portable_cls::Result classify(const cv::Mat& image, int topk) = 0;
    virtual std::string backend() const = 0;
};

#if defined(VISION_APP_HAS_ONNX)
class OnnxRuntimeImpl final : public RuntimeIface {
public:
    explicit OnnxRuntimeImpl(const ModelConfig& cfg) {
        portable_cls::OnnxClassifier::Config c;
        c.input_width = cfg.input_width;
        c.input_height = cfg.input_height;
        c.topk = cfg.topk;
        c.num_threads = cfg.threads;
        c.preprocess = portable_cls::preprocess_mode_from_string(cfg.preprocess);
        c.suppress_stdio_on_load = true;
        clf_.load(cfg.onnx_path, cfg.labels_path, c);
    }
    portable_cls::Result classify(const cv::Mat& image, int topk) override { return clf_.classify(image, topk); }
    std::string backend() const override { return "onnx"; }
private:
    portable_cls::OnnxClassifier clf_;
};
#endif

#if defined(VISION_APP_HAS_NCNN)
class NcnnRuntimeImpl final : public RuntimeIface {
public:
    explicit NcnnRuntimeImpl(const ModelConfig& cfg) {
        portable_cls::NcnnClassifier::Config c;
        c.input_width = cfg.input_width;
        c.input_height = cfg.input_height;
        c.topk = cfg.topk;
        c.num_threads = cfg.threads;
        c.preprocess = portable_cls::preprocess_mode_from_string(cfg.preprocess);
        c.use_vulkan = false;
        clf_.load(cfg.ncnn_param_path, cfg.ncnn_bin_path, cfg.labels_path, c);
    }
    portable_cls::Result classify(const cv::Mat& image, int topk) override { return clf_.classify(image, topk); }
    std::string backend() const override { return "ncnn"; }
private:
    portable_cls::NcnnClassifier clf_;
};
#endif

std::unique_ptr<RuntimeIface> g_runtime;
std::string g_backend = "off";

} // namespace

bool init_model_runtime(const ModelConfig& cfg, std::string& err) {
    release_model_runtime();
    err.clear();
    if (!cfg.enable || cfg.backend == "off") return false;
    try {
        if (cfg.backend == "onnx") {
#if defined(VISION_APP_HAS_ONNX)
            if (cfg.onnx_path.empty()) throw std::runtime_error("model_onnx_path is empty");
            if (cfg.labels_path.empty()) throw std::runtime_error("model_labels_path is empty");
            g_runtime = std::make_unique<OnnxRuntimeImpl>(cfg);
            g_backend = "onnx";
            return true;
#else
            throw std::runtime_error("ONNX backend not built in this binary");
#endif
        }
        if (cfg.backend == "ncnn") {
#if defined(VISION_APP_HAS_NCNN)
            if (cfg.ncnn_param_path.empty()) throw std::runtime_error("model_ncnn_param_path is empty");
            if (cfg.ncnn_bin_path.empty()) throw std::runtime_error("model_ncnn_bin_path is empty");
            if (cfg.labels_path.empty()) throw std::runtime_error("model_labels_path is empty");
            g_runtime = std::make_unique<NcnnRuntimeImpl>(cfg);
            g_backend = "ncnn";
            return true;
#else
            throw std::runtime_error("NCNN backend not built in this binary");
#endif
        }
        throw std::runtime_error("unknown backend: " + cfg.backend);
    } catch (const std::exception& e) {
        err = e.what();
        release_model_runtime();
        return false;
    }
}

void release_model_runtime() {
    g_runtime.reset();
    g_backend = "off";
}

bool model_runtime_ready() {
    return static_cast<bool>(g_runtime);
}

std::string model_runtime_backend() {
    return g_backend;
}

bool run_model_on_image_roi(const RoiRuntimeData& in,
                            const ModelConfig& cfg,
                            ModelResult& out,
                            std::string& err) {
    out = {};
    out.ran = true;
    if (!g_runtime) {
        err = "model not loaded";
        out.summary = "model off";
        return false;
    }
    if (in.image_bgr.empty()) {
        err = "empty image_roi";
        out.summary = "empty roi";
        return false;
    }
    try {
        auto t0 = std::chrono::steady_clock::now();
        auto r = g_runtime->classify(in.image_bgr, cfg.topk);
        auto t1 = std::chrono::steady_clock::now();
        out.infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        out.ok = true;
        out.top_index = r.best_index;
        out.top_score = r.best_probability;
        out.top_label = r.best_label;
        out.topk = {};
        for (const auto& s : r.topk) out.topk.push_back({s.index, s.probability, s.label});
        out.summary = g_backend + " " + out.top_label + " " + std::to_string(out.top_score).substr(0,6) +
                      " " + std::to_string(out.infer_ms).substr(0,6) + "ms";
        err.clear();
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        out.summary = "model err";
        return false;
    }
}

} // namespace vision_app
