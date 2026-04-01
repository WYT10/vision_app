
#include "model.hpp"

#include <chrono>
#include <memory>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#include "classifier_common.hpp"

#if defined(VISION_APP_HAS_ONNX)
#include "onnx_classifier.hpp"
#endif
#if defined(VISION_APP_HAS_NCNN)
#include "ncnn_classifier.hpp"
#endif

namespace vision_app {
namespace {

enum class BackendKind { Off, Onnx, Ncnn };

BackendKind g_backend = BackendKind::Off;
#if defined(VISION_APP_HAS_ONNX)
std::unique_ptr<portable_cls::OnnxClassifier> g_onnx;
#endif
#if defined(VISION_APP_HAS_NCNN)
std::unique_ptr<portable_cls::NcnnClassifier> g_ncnn;
#endif

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

void release_model_runtime() {
#if defined(VISION_APP_HAS_ONNX)
    g_onnx.reset();
#endif
#if defined(VISION_APP_HAS_NCNN)
    g_ncnn.reset();
#endif
    g_backend = BackendKind::Off;
}

bool init_model_runtime(const ModelConfig& cfg, std::string& err) {
    release_model_runtime();
    err.clear();
    if (!cfg.enable || cfg.backend == "off") return false;

    try {
        if (cfg.labels_path.empty()) throw std::runtime_error("labels path is empty");

        if (cfg.backend == "onnx") {
#if defined(VISION_APP_HAS_ONNX)
            if (cfg.onnx_path.empty()) throw std::runtime_error("onnx path is empty");
            auto clf = std::make_unique<portable_cls::OnnxClassifier>();
            portable_cls::OnnxClassifier::Config c;
            c.input_width = cfg.input_width;
            c.input_height = cfg.input_height;
            c.topk = cfg.topk;
            c.num_threads = cfg.threads;
            c.preprocess = parse_preprocess(cfg.preprocess);
            c.suppress_stdio_on_load = true;
            clf->load(cfg.onnx_path, cfg.labels_path, c);
            g_onnx = std::move(clf);
            g_backend = BackendKind::Onnx;
            return true;
#else
            throw std::runtime_error("vision_app built without ONNX Runtime support");
#endif
        }

        if (cfg.backend == "ncnn") {
#if defined(VISION_APP_HAS_NCNN)
            if (cfg.ncnn_param_path.empty() || cfg.ncnn_bin_path.empty()) {
                throw std::runtime_error("ncnn param/bin paths are empty");
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
            g_backend = BackendKind::Ncnn;
            return true;
#else
            throw std::runtime_error("vision_app built without NCNN support");
#endif
        }

        throw std::runtime_error("unsupported backend: " + cfg.backend);
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

    try {
        const cv::Mat clean = apply_mask_fill(in.image_bgr, in.image_mask, cv::Scalar(255,255,255));

        auto t0 = std::chrono::steady_clock::now();
        portable_cls::Result r;
        if (g_backend == BackendKind::Onnx) {
#if defined(VISION_APP_HAS_ONNX)
            if (!g_onnx) throw std::runtime_error("onnx model not loaded");
            r = g_onnx->classify(clean, cfg.topk);
#else
            throw std::runtime_error("onnx backend unavailable");
#endif
        } else if (g_backend == BackendKind::Ncnn) {
#if defined(VISION_APP_HAS_NCNN)
            if (!g_ncnn) throw std::runtime_error("ncnn model not loaded");
            r = g_ncnn->classify(clean, cfg.topk);
#else
            throw std::runtime_error("ncnn backend unavailable");
#endif
        } else {
            throw std::runtime_error("model backend not initialized");
        }
        out.infer_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        out.ok = true;
        out.best.index = r.best_index;
        out.best.score = r.best_probability;
        out.best.label = r.best_label;
        out.topk.reserve(r.topk.size());
        for (const auto& s : r.topk) out.topk.push_back({s.index, s.probability, s.label});

        std::ostringstream oss;
        oss << cfg.backend << " " << out.best.label << " "
            << std::fixed << std::setprecision(4) << out.best.score << " "
            << std::setprecision(3) << out.infer_ms << "ms";
        out.summary = oss.str();
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        out.ok = false;
        out.summary = std::string("model err: ") + err;
        return false;
    }
}

} // namespace vision_app
