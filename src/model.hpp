#pragma once
#include <string>
#include "calibrate.hpp"

namespace vision_app {

struct ModelHit {
    int index = -1;
    float score = 0.f;
    std::string label;
};

struct ModelConfig {
    bool enable = false;
    std::string backend = "off"; // off | onnx | ncnn
    std::string onnx_path;
    std::string ncnn_param_path;
    std::string ncnn_bin_path;
    std::string labels_path;
    int input_width = 128;
    int input_height = 128;
    std::string preprocess = "crop"; // crop | stretch | letterbox
    int threads = 4;
    int stride = 1;
    int topk = 5;
};

struct ModelResult {
    bool ran = false;
    bool ok = false;
    ModelHit best;
    std::vector<ModelHit> topk;
    double infer_ms = 0.0;
    std::string summary;
};

struct RoiRuntimeData {
    cv::Mat red_bgr;
    cv::Mat red_mask;
    cv::Mat image_bgr;
    cv::Mat image_mask;
    cv::Mat red_mask_vis;

    std::string runtime_mode = "fixed";
    cv::Rect fixed_red_rect;
    cv::Rect fixed_image_rect;
    cv::Rect dynamic_search_rect;
    cv::Rect dynamic_image_rect;

    double red_ratio = 0.0;
    int red_valid_pixels = 0;
    int image_valid_pixels = 0;
    int red_center_x = -1;
    int red_blob_area = 0;
    bool red_found = false;
    bool used_last_center = false;
    bool used_fallback_center = false;
};

bool init_model_runtime(const ModelConfig& cfg, std::string& err);
void release_model_runtime();
bool run_model_on_image_roi(const RoiRuntimeData& in,
                            const ModelConfig& cfg,
                            ModelResult& out,
                            std::string& err);

} // namespace vision_app
