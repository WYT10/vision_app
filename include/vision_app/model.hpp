
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
    cv::Rect image_roi_rect{};
    cv::Rect upper_zone_rect{};
    cv::Rect lower_zone_rect{};
    double red_ratio = 0.0;
    int red_valid_pixels = 0;
    int image_valid_pixels = 0;
    bool trigger_ready = false;
    bool upper_valid = false;
    bool lower_valid = false;
    int upper_red_pixels = 0;
    int lower_red_pixels = 0;
    double upper_red_ratio = 0.0;
    double lower_red_ratio = 0.0;
    double x_upper = -1.0;
    double x_lower = -1.0;
    double x_center = -1.0;
};

bool init_model_runtime(const ModelConfig& cfg, std::string& err);
void release_model_runtime();
bool run_model_on_image_roi(const RoiRuntimeData& in,
                            const ModelConfig& cfg,
                            ModelResult& out,
                            std::string& err);

} // namespace vision_app
