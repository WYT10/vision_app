#include "deploy.hpp"

#include <cmath>
#include <opencv2/dnn.hpp>

namespace vision_app {
namespace {
cv::dnn::Net g_net;
ModelConfig g_cfg;
bool g_loaded = false;

cv::Mat apply_mask_fill(const cv::Mat& img, const cv::Mat& mask, const cv::Scalar& fill) {
    cv::Mat out = img.clone();
    if (out.empty() || mask.empty()) return out;
    cv::Mat inv;
    cv::bitwise_not(mask, inv);
    out.setTo(fill, inv);
    return out;
}

std::string backend_name(const std::string& s) {
    if (s == "onnx") return "onnx";
    if (s == "ncnn") return "ncnn";
    return "off";
}
} // namespace

bool init_model_runtime(const ModelConfig& cfg, std::string& err) {
    release_model_runtime();
    g_cfg = cfg;
    if (!cfg.enable || cfg.backend == "off" || cfg.path.empty()) {
        err.clear();
        return false;
    }
    if (backend_name(cfg.backend) == "onnx") {
        try {
            g_net = cv::dnn::readNet(cfg.path);
            if (g_net.empty()) {
                err = "failed to load onnx model: " + cfg.path;
                return false;
            }
            g_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            g_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            g_loaded = true;
            err.clear();
            return true;
        } catch (const cv::Exception& e) {
            err = e.what();
            return false;
        }
    }
    err = "NCNN backend hook not compiled in this drop; use --model-backend onnx for now";
    return false;
}

void release_model_runtime() {
    g_net = cv::dnn::Net();
    g_loaded = false;
}

bool extract_runtime_rois(const cv::Mat& warped,
                          const cv::Mat& valid_mask,
                          const RoiConfig& rois,
                          const RedThresholdConfig& red_cfg,
                          RoiRuntimeData& out,
                          std::string& err) {
    out = {};
    if (warped.empty() || valid_mask.empty()) {
        err = "empty warped image or mask";
        return false;
    }
    const cv::Rect rr = roi_to_rect(rois.red_roi, warped.size());
    const cv::Rect ir = roi_to_rect(rois.image_roi, warped.size());
    out.red_bgr = warped(rr).clone();
    out.red_mask = valid_mask(rr).clone();
    out.image_bgr = warped(ir).clone();
    out.image_mask = valid_mask(ir).clone();
    out.red_valid_pixels = cv::countNonZero(out.red_mask);
    out.image_valid_pixels = cv::countNonZero(out.image_mask);

    cv::Mat hsv;
    cv::cvtColor(out.red_bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat m1, m2, red_mask;
    cv::inRange(hsv,
                cv::Scalar(red_cfg.h1_low, red_cfg.s_min, red_cfg.v_min),
                cv::Scalar(red_cfg.h1_high, 255, 255),
                m1);
    cv::inRange(hsv,
                cv::Scalar(red_cfg.h2_low, red_cfg.s_min, red_cfg.v_min),
                cv::Scalar(red_cfg.h2_high, 255, 255),
                m2);
    cv::bitwise_or(m1, m2, red_mask);
    cv::bitwise_and(red_mask, out.red_mask, red_mask);

    const int valid = std::max(1, out.red_valid_pixels);
    out.red_ratio = static_cast<double>(cv::countNonZero(red_mask)) / static_cast<double>(valid);
    err.clear();
    return true;
}

bool run_model_on_image_roi(const RoiRuntimeData& in,
                            const ModelConfig& cfg,
                            ModelResult& out,
                            std::string& err) {
    out = {};
    out.ran = true;
    if (!g_loaded || backend_name(cfg.backend) != "onnx") {
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
        cv::Mat clean = apply_mask_fill(in.image_bgr, in.image_mask, cv::Scalar(255,255,255));
        cv::Mat blob = cv::dnn::blobFromImage(clean,
                                              cfg.scale,
                                              cv::Size(cfg.input_width, cfg.input_height),
                                              cv::Scalar(),
                                              cfg.swap_rb,
                                              false,
                                              CV_32F);
        g_net.setInput(blob);
        cv::Mat out_blob = g_net.forward();
        cv::Mat flat = out_blob.reshape(1, 1);
        if (flat.cols <= 0) {
            err = "empty model output";
            out.summary = "empty output";
            return false;
        }
        cv::Point max_loc;
        double max_val = 0.0;
        cv::minMaxLoc(flat, nullptr, &max_val, nullptr, &max_loc);
        out.ok = true;
        out.top_index = max_loc.x;
        out.top_score = static_cast<float>(max_val);
        out.summary = "onnx top1=" + std::to_string(out.top_index) + " score=" + std::to_string(out.top_score).substr(0,6);
        err.clear();
        return true;
    } catch (const cv::Exception& e) {
        err = e.what();
        out.summary = "model err";
        return false;
    }
}

} // namespace vision_app
