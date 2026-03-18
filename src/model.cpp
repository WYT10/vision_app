#include "deploy.hpp"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

#include "classifier_common.hpp"

#if defined(HAVE_ONNX_RUNTIME)
#include "onnx_classifier.hpp"
#endif
#if defined(HAVE_NCNN)
#include "ncnn_classifier.hpp"
#endif

namespace vision_app {
namespace fs = std::filesystem;

namespace {

class IModelRuntime {
public:
    virtual ~IModelRuntime() = default;
    virtual portable_cls::Result classify(const cv::Mat& image_bgr, int topk) const = 0;
};

#if defined(HAVE_ONNX_RUNTIME)
class OnnxRuntimeAdapter final : public IModelRuntime {
public:
    explicit OnnxRuntimeAdapter(std::unique_ptr<portable_cls::OnnxClassifier> impl)
        : impl_(std::move(impl)) {}

    portable_cls::Result classify(const cv::Mat& image_bgr, int topk) const override {
        return impl_->classify(image_bgr, topk);
    }

private:
    std::unique_ptr<portable_cls::OnnxClassifier> impl_;
};
#endif

#if defined(HAVE_NCNN)
class NcnnRuntimeAdapter final : public IModelRuntime {
public:
    explicit NcnnRuntimeAdapter(std::unique_ptr<portable_cls::NcnnClassifier> impl)
        : impl_(std::move(impl)) {}

    portable_cls::Result classify(const cv::Mat& image_bgr, int topk) const override {
        return impl_->classify(image_bgr, topk);
    }

private:
    std::unique_ptr<portable_cls::NcnnClassifier> impl_;
};
#endif

std::unique_ptr<IModelRuntime> g_runtime;
ModelConfig g_cfg;
bool g_loaded = false;

std::string backend_name(const std::string& s) {
    if (s == "onnx") return "onnx";
    if (s == "ncnn") return "ncnn";
    return "off";
}

cv::Mat apply_mask_fill(const cv::Mat& img, const cv::Mat& mask, const cv::Scalar& fill) {
    cv::Mat out = img.clone();
    if (out.empty() || mask.empty()) return out;
    cv::Mat inv;
    cv::bitwise_not(mask, inv);
    out.setTo(fill, inv);
    return out;
}

portable_cls::PreprocessMode preprocess_mode_from_string(const std::string& s) {
    return portable_cls::preprocess_mode_from_string(s.empty() ? std::string("crop") : s);
}

portable_cls::CommonConfig build_common_cfg(const ModelConfig& cfg) {
    portable_cls::CommonConfig c;
    c.input_width = std::max(1, cfg.input_width);
    c.input_height = std::max(1, cfg.input_height);
    c.topk = std::max(1, cfg.topk);
    c.num_threads = std::max(1, cfg.threads);
    c.preprocess = preprocess_mode_from_string(cfg.preprocess);
    c.mean_vals = cfg.mean_vals;
    c.norm_vals = cfg.norm_vals;
    return c;
}

bool file_exists(const std::string& path) {
    return !path.empty() && fs::exists(fs::path(path));
}

std::string resolve_onnx_path(const ModelConfig& cfg) {
    if (file_exists(cfg.onnx_path)) return cfg.onnx_path;
    if (backend_name(cfg.backend) == "onnx" && file_exists(cfg.path)) return cfg.path;
    return {};
}

std::pair<std::string, std::string> resolve_ncnn_paths(const ModelConfig& cfg) {
    std::string param = file_exists(cfg.ncnn_param_path) ? cfg.ncnn_param_path : std::string();
    std::string bin = file_exists(cfg.ncnn_bin_path) ? cfg.ncnn_bin_path : std::string();

    if (backend_name(cfg.backend) == "ncnn" && param.empty() && file_exists(cfg.path)) {
        fs::path p(cfg.path);
        if (p.extension() == ".param") {
            param = cfg.path;
            fs::path guessed = p;
            guessed.replace_extension(".bin");
            if (bin.empty() && fs::exists(guessed)) bin = guessed.string();
        }
    }
    return {param, bin};
}

std::string label_or_index(const std::string& label, int idx) {
    if (!label.empty()) return label;
    return std::string("class_") + std::to_string(idx);
}

}  // namespace

bool init_model_runtime(const ModelConfig& cfg, std::string& err) {
    release_model_runtime();
    g_cfg = cfg;

    if (!cfg.enable || backend_name(cfg.backend) == "off") {
        err.clear();
        return false;
    }
    if (!file_exists(cfg.labels_path)) {
        err = "labels file not found: " + cfg.labels_path;
        return false;
    }

    try {
        if (backend_name(cfg.backend) == "onnx") {
            const std::string model_path = resolve_onnx_path(cfg);
            if (model_path.empty()) {
                err = "onnx model not found; set model_onnx_path or model_path";
                return false;
            }
#if defined(HAVE_ONNX_RUNTIME)
            auto clf = std::make_unique<portable_cls::OnnxClassifier>();
            portable_cls::OnnxClassifier::Config ocfg;
            const auto ccfg = build_common_cfg(cfg);
            ocfg.input_width = ccfg.input_width;
            ocfg.input_height = ccfg.input_height;
            ocfg.topk = ccfg.topk;
            ocfg.num_threads = ccfg.num_threads;
            ocfg.preprocess = ccfg.preprocess;
            ocfg.mean_vals = ccfg.mean_vals;
            ocfg.norm_vals = ccfg.norm_vals;
            ocfg.suppress_stdio_on_load = cfg.quiet_onnx_load;
            clf->load(model_path, cfg.labels_path, ocfg);
            g_runtime = std::make_unique<OnnxRuntimeAdapter>(std::move(clf));
            g_loaded = true;
            err.clear();
            return true;
#else
            err = "onnx backend not compiled; rebuild with ENABLE_ONNX_RUNTIME=ON";
            return false;
#endif
        }

        if (backend_name(cfg.backend) == "ncnn") {
            const auto [param_path, bin_path] = resolve_ncnn_paths(cfg);
            if (param_path.empty() || bin_path.empty()) {
                err = "ncnn model not found; set model_ncnn_param_path/model_ncnn_bin_path";
                return false;
            }
#if defined(HAVE_NCNN)
            auto clf = std::make_unique<portable_cls::NcnnClassifier>();
            portable_cls::NcnnClassifier::Config ncfg;
            const auto ccfg = build_common_cfg(cfg);
            ncfg.input_width = ccfg.input_width;
            ncfg.input_height = ccfg.input_height;
            ncfg.topk = ccfg.topk;
            ncfg.num_threads = ccfg.num_threads;
            ncfg.preprocess = ccfg.preprocess;
            ncfg.mean_vals = ccfg.mean_vals;
            ncfg.norm_vals = ccfg.norm_vals;
            ncfg.use_vulkan = false;
            clf->load(param_path, bin_path, cfg.labels_path, ncfg);
            g_runtime = std::make_unique<NcnnRuntimeAdapter>(std::move(clf));
            g_loaded = true;
            err.clear();
            return true;
#else
            err = "ncnn backend not compiled; rebuild with ENABLE_NCNN=ON";
            return false;
#endif
        }

        err = "unsupported backend: " + cfg.backend;
        return false;
    } catch (const std::exception& e) {
        release_model_runtime();
        err = e.what();
        return false;
    }
}

void release_model_runtime() {
    g_runtime.reset();
    g_loaded = false;
}

bool run_model_on_image_roi(const RoiRuntimeData& in,
                            const ModelConfig& cfg,
                            ModelResult& out,
                            std::string& err) {
    out = {};
    out.ran = true;

    if (!g_loaded || !g_runtime) {
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
        cv::Mat clean = apply_mask_fill(in.image_bgr, in.image_mask, cv::Scalar(255, 255, 255));
        const portable_cls::Result r = g_runtime->classify(clean, std::max(1, cfg.topk));
        out.ok = !r.topk.empty();
        out.top_index = r.best_index;
        out.top_score = r.best_probability;
        out.top_label = label_or_index(r.best_label, r.best_index);
        out.summary = cfg.backend + " " + out.top_label + " " + std::to_string(out.top_score).substr(0, 6);
        out.topk.clear();
        for (const auto& s : r.topk) {
            out.topk.push_back({s.index, s.probability, label_or_index(s.label, s.index)});
        }
        err.clear();
        return out.ok;
    } catch (const std::exception& e) {
        err = e.what();
        out.summary = cfg.backend + " err";
        return false;
    }
}

}  // namespace vision_app
