
#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace vision_app {

struct RedThresholdConfig {
    int h1_low = 0;
    int h1_high = 10;
    int h2_low = 170;
    int h2_high = 180;
    int s_min = 80;
    int v_min = 60;
};

struct ModelConfig {
    bool enable = false;
    std::string backend = "off";   // off | onnx | ncnn
    std::string onnx_path;
    std::string ncnn_param_path;
    std::string ncnn_bin_path;
    std::string labels_path;
    int input_width = 128;
    int input_height = 128;
    std::string preprocess = "crop"; // crop | stretch
    int threads = 4;
    int stride = 5;
    int topk = 5;
};

struct ModelHit {
    int index = -1;
    float score = 0.f;
    std::string label;
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
    double red_ratio = 0.0;
    int red_valid_pixels = 0;
    int image_valid_pixels = 0;
};

bool init_model_runtime(const ModelConfig& cfg, std::string& err);
void release_model_runtime();
bool run_model_on_image_roi(const RoiRuntimeData& in,
                            const ModelConfig& cfg,
                            ModelResult& out,
                            std::string& err);

} // namespace vision_app
