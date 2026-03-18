
#include "model.hpp"

#include <memory>

#include "classifier_common.hpp"

#if defined(VISION_APP_HAS_ONNX)
#include "onnx_classifier.hpp"
#endif
#if defined(VISION_APP_HAS_NCNN)
#include "ncnn_classifier.hpp"
#endif

namespace vision_app {
namespace {
struct Runtime {
    std::string backend = "off";
#if defined(VISION_APP_HAS_ONNX)
    std::unique_ptr<OnnxClassifier> onnx;
#endif
#if defined(VISION_APP_HAS_NCNN)
    std::unique_ptr<NcnnClassifier> ncnn;
#endif
};
Runtime g_rt;
}

bool init_model_runtime(const ModelConfig& cfg, std::string& err) {
    release_model_runtime();
    if (!cfg.enable || cfg.backend == "off") { err.clear(); return false; }
    if (cfg.backend == "onnx") {
#if defined(VISION_APP_HAS_ONNX)
        if (cfg.onnx_path.empty()) { err = "onnx path empty"; return false; }
        if (cfg.labels_path.empty()) { err = "labels path empty"; return false; }
        auto clf = std::make_unique<OnnxClassifier>();
        OnnxClassifier::Config cc;
        cc.input_width = cfg.input_width;
        cc.input_height = cfg.input_height;
        cc.preprocess = cfg.preprocess;
        cc.topk = cfg.topk;
        if (!clf->load(cfg.onnx_path, cfg.labels_path, cc, err)) return false;
        g_rt.backend = "onnx";
        g_rt.onnx = std::move(clf);
        return true;
#else
        err = "onnx runtime not built";
        return false;
#endif
    }
    if (cfg.backend == "ncnn") {
#if defined(VISION_APP_HAS_NCNN)
        if (cfg.ncnn_param_path.empty() || cfg.ncnn_bin_path.empty()) { err = "ncnn param/bin path empty"; return false; }
        if (cfg.labels_path.empty()) { err = "labels path empty"; return false; }
        auto clf = std::make_unique<NcnnClassifier>();
        NcnnClassifier::Config cc;
        cc.input_width = cfg.input_width;
        cc.input_height = cfg.input_height;
        cc.preprocess = cfg.preprocess;
        cc.topk = cfg.topk;
        cc.threads = cfg.threads;
        if (!clf->load(cfg.ncnn_param_path, cfg.ncnn_bin_path, cfg.labels_path, cc, err)) return false;
        g_rt.backend = "ncnn";
        g_rt.ncnn = std::move(clf);
        return true;
#else
        err = "ncnn runtime not built";
        return false;
#endif
    }
    err = "unknown backend: " + cfg.backend;
    return false;
}

void release_model_runtime() {
    g_rt = {};
}

bool run_model_on_image_roi(const RoiRuntimeData& in,
                            const ModelConfig& cfg,
                            ModelResult& out,
                            std::string& err) {
    out = {};
    out.ran = true;
    if (!cfg.enable || cfg.backend == "off") {
        err = "model disabled";
        out.summary = "model off";
        return false;
    }
    if (in.image_bgr.empty()) {
        err = "empty image_roi";
        out.summary = "empty roi";
        return false;
    }
    const cv::Mat clean = in.image_bgr.clone();
    if (g_rt.backend == "onnx") {
#if defined(VISION_APP_HAS_ONNX)
        if (!g_rt.onnx) { err = "onnx model not loaded"; out.summary = err; return false; }
        auto r = g_rt.onnx->classify(clean);
        out.ok = r.ok;
        out.infer_ms = r.infer_ms;
        out.summary = r.summary;
        for (const auto& h : r.topk) out.topk.push_back({h.index, h.score, h.label});
        if (!out.topk.empty()) out.best = out.topk.front();
        err = r.ok ? std::string() : r.summary;
        return r.ok;
#else
        err = "onnx runtime not built";
        out.summary = err;
        return false;
#endif
    }
    if (g_rt.backend == "ncnn") {
#if defined(VISION_APP_HAS_NCNN)
        if (!g_rt.ncnn) { err = "ncnn model not loaded"; out.summary = err; return false; }
        auto r = g_rt.ncnn->classify(clean);
        out.ok = r.ok;
        out.infer_ms = r.infer_ms;
        out.summary = r.summary;
        for (const auto& h : r.topk) out.topk.push_back({h.index, h.score, h.label});
        if (!out.topk.empty()) out.best = out.topk.front();
        err = r.ok ? std::string() : r.summary;
        return r.ok;
#else
        err = "ncnn runtime not built";
        out.summary = err;
        return false;
#endif
    }
    err = "unknown runtime backend";
    out.summary = err;
    return false;
}

} // namespace vision_app
