#include "model.hpp"

#include <chrono>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "classifier_common.hpp"
#include "ncnn_classifier.hpp"

namespace vision_app {
namespace {

std::unique_ptr<portable_cls::NcnnClassifier> g_ncnn;

portable_cls::PreprocessMode parse_preprocess(const std::string& s) {
    if (s == "stretch") return portable_cls::PreprocessMode::ResizeStretch;
    if (s == "letterbox") return portable_cls::PreprocessMode::LetterboxSquare;
    return portable_cls::PreprocessMode::CenterCropSquare;
}

cv::Mat apply_mask_fill(const cv::Mat& img, const cv::Mat& mask, const cv::Scalar& fill) {
    cv::Mat out = img.clone();
    if (out.empty() || mask.empty()) return out;
    cv::Mat inv;
    cv::bitwise_not(mask, inv);
    out.setTo(fill, inv);
    return out;
}

} // namespace

void release_model_runtime() { g_ncnn.reset(); }

bool init_model_runtime(const ModelConfig& cfg, std::string& err) {
    release_model_runtime();
    err.clear();
    if (!cfg.enable || cfg.backend == "off") return false;
    if (cfg.backend != "ncnn") {
        err = "unsupported backend: " + cfg.backend + " (Pi runtime is NCNN-only)";
        return false;
    }

    try {
        if (cfg.ncnn_param_path.empty() || cfg.ncnn_bin_path.empty()) {
            throw std::runtime_error("ncnn param/bin paths are empty");
        }
        if (cfg.labels_path.empty()) {
            throw std::runtime_error("labels path is empty");
        }

        auto clf = std::make_unique<portable_cls::NcnnClassifier>();
        portable_cls::NcnnClassifier::Config c;
        c.input_width = cfg.input_width;
        c.input_height = cfg.input_height;
        c.topk = cfg.topk;
        c.num_threads = cfg.threads;
        c.preprocess = parse_preprocess(cfg.preprocess);
        c.use_vulkan = false;
        c.auto_detect_blob_names = true;
        clf->load(cfg.ncnn_param_path, cfg.ncnn_bin_path, cfg.labels_path, c);
        g_ncnn = std::move(clf);
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        release_model_runtime();
        return false;
    }
}

bool run_model_on_image_roi(const RoiRuntimeData& in,
                            const ModelConfig& cfg,
                            ModelResult& out,
                            std::string& err) {
    out = {};
    out.ran = true;
    err.clear();

    if (in.image_bgr.empty()) {
        err = "empty image_roi";
        out.summary = "empty roi";
        return false;
    }
    if (!g_ncnn) {
        err = "ncnn model not loaded";
        out.summary = "model err: " + err;
        return false;
    }

    try {
        const cv::Mat clean = apply_mask_fill(in.image_bgr, in.image_mask, cv::Scalar(255, 255, 255));
        const auto t0 = std::chrono::steady_clock::now();
        const portable_cls::Result r = g_ncnn->classify(clean, cfg.topk);
        out.infer_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        out.ok = true;
        out.best.index = r.best_index;
        out.best.score = r.best_probability;
        out.best.label = r.best_label;
        out.topk.reserve(r.topk.size());
        for (const auto& s : r.topk) out.topk.push_back({s.index, s.probability, s.label});

        std::ostringstream oss;
        oss << "ncnn " << out.best.label << ' ' << std::fixed << std::setprecision(4) << out.best.score
            << ' ' << std::setprecision(3) << out.infer_ms << "ms";
        out.summary = oss.str();
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        out.ok = false;
        out.summary = "model err: " + err;
        return false;
    }
}

} // namespace vision_app
