#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace portable_cls {
namespace fs = std::filesystem;

enum class PreprocessMode {
    CenterCropSquare,
    ResizeStretch,
    LetterboxSquare,
};

struct CommonConfig {
    int input_width = 224;
    int input_height = 224;
    int topk = 5;
    int num_threads = 4;
    PreprocessMode preprocess = PreprocessMode::CenterCropSquare;
    std::array<float, 3> mean_vals{0.f, 0.f, 0.f};
    std::array<float, 3> norm_vals{1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
};

struct Score {
    int index = -1;
    float probability = 0.f;
    std::string label;
};

struct Result {
    std::vector<float> probabilities;
    std::vector<Score> topk;
    int best_index = -1;
    float best_probability = 0.f;
    std::string best_label;
};

inline std::vector<std::string> read_labels(const fs::path& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("Failed to open labels file: " + path.string());
    }

    std::vector<std::string> labels;
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            labels.push_back(line);
        }
    }
    if (labels.empty()) {
        throw std::runtime_error("Labels file is empty: " + path.string());
    }
    return labels;
}

inline cv::Mat ensure_bgr(const cv::Mat& src) {
    if (src.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    if (src.channels() == 3) {
        return src;
    }
    cv::Mat dst;
    if (src.channels() == 1) {
        cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);
        return dst;
    }
    if (src.channels() == 4) {
        cv::cvtColor(src, dst, cv::COLOR_BGRA2BGR);
        return dst;
    }
    throw std::runtime_error("Unsupported channel count: " + std::to_string(src.channels()));
}

inline cv::Scalar estimate_border_color(const cv::Mat& img) {
    if (img.empty()) {
        return cv::Scalar(114, 114, 114);
    }
    const int patch = std::max(1, std::min(img.cols, img.rows) / 12);
    cv::Rect tl(0, 0, patch, patch);
    cv::Rect tr(std::max(0, img.cols - patch), 0, patch, patch);
    cv::Rect bl(0, std::max(0, img.rows - patch), patch, patch);
    cv::Rect br(std::max(0, img.cols - patch), std::max(0, img.rows - patch), patch, patch);
    const cv::Scalar a = cv::mean(img(tl));
    const cv::Scalar b = cv::mean(img(tr));
    const cv::Scalar c = cv::mean(img(bl));
    const cv::Scalar d = cv::mean(img(br));
    return cv::Scalar((a[0] + b[0] + c[0] + d[0]) * 0.25,
                      (a[1] + b[1] + c[1] + d[1]) * 0.25,
                      (a[2] + b[2] + c[2] + d[2]) * 0.25);
}

inline cv::Mat resize_stretch(const cv::Mat& img, int out_w, int out_h) {
    cv::Mat out;
    cv::resize(img, out, cv::Size(out_w, out_h), 0.0, 0.0, cv::INTER_LINEAR);
    return out;
}

inline cv::Mat center_crop_square_resize(const cv::Mat& img, int out_w, int out_h) {
    const int side = std::min(img.cols, img.rows);
    const int x = (img.cols - side) / 2;
    const int y = (img.rows - side) / 2;
    const cv::Rect roi(x, y, side, side);
    return resize_stretch(img(roi), out_w, out_h);
}

inline cv::Mat letterbox_square_resize(const cv::Mat& img, int out_w, int out_h) {
    const int side = std::max(img.cols, img.rows);
    cv::Mat canvas(side, side, img.type(), estimate_border_color(img));
    const int x = (side - img.cols) / 2;
    const int y = (side - img.rows) / 2;
    img.copyTo(canvas(cv::Rect(x, y, img.cols, img.rows)));
    return resize_stretch(canvas, out_w, out_h);
}

inline cv::Mat preprocess_image(const cv::Mat& src_bgr, const CommonConfig& cfg) {
    const cv::Mat bgr = ensure_bgr(src_bgr);
    switch (cfg.preprocess) {
        case PreprocessMode::CenterCropSquare:
            return center_crop_square_resize(bgr, cfg.input_width, cfg.input_height);
        case PreprocessMode::ResizeStretch:
            return resize_stretch(bgr, cfg.input_width, cfg.input_height);
        case PreprocessMode::LetterboxSquare:
            return letterbox_square_resize(bgr, cfg.input_width, cfg.input_height);
    }
    return center_crop_square_resize(bgr, cfg.input_width, cfg.input_height);
}

inline std::vector<float> softmax(const std::vector<float>& logits) {
    if (logits.empty()) {
        return {};
    }
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> out(logits.size());
    float sum = 0.f;
    for (size_t i = 0; i < logits.size(); ++i) {
        out[i] = std::exp(logits[i] - max_logit);
        sum += out[i];
    }
    if (sum > 0.f) {
        for (float& v : out) {
            v /= sum;
        }
    }
    return out;
}

inline bool looks_like_probability_distribution(const std::vector<float>& values) {
    if (values.empty()) {
        return false;
    }
    float sum = 0.f;
    for (float v : values) {
        if (v < -1e-6f || v > 1.0001f || !std::isfinite(v)) {
            return false;
        }
        sum += v;
    }
    return std::abs(sum - 1.f) < 1e-3f;
}

inline std::vector<float> probabilities_from_output(const std::vector<float>& output) {
    if (looks_like_probability_distribution(output)) {
        return output;
    }
    return softmax(output);
}

inline std::vector<Score> select_topk(const std::vector<float>& probabilities,
                                      const std::vector<std::string>& labels,
                                      int k) {
    if (probabilities.empty()) {
        return {};
    }
    std::vector<int> indices(probabilities.size());
    std::iota(indices.begin(), indices.end(), 0);
    const int use_k = std::max(1, std::min<int>(k, static_cast<int>(indices.size())));
    std::partial_sort(indices.begin(), indices.begin() + use_k, indices.end(),
                      [&](int a, int b) { return probabilities[a] > probabilities[b]; });

    std::vector<Score> out;
    out.reserve(use_k);
    for (int i = 0; i < use_k; ++i) {
        const int idx = indices[i];
        Score s;
        s.index = idx;
        s.probability = probabilities[idx];
        if (idx >= 0 && idx < static_cast<int>(labels.size())) {
            s.label = labels[idx];
        } else {
            s.label = "class_" + std::to_string(idx);
        }
        out.push_back(std::move(s));
    }
    return out;
}

inline std::string preprocess_mode_to_string(PreprocessMode mode) {
    switch (mode) {
        case PreprocessMode::CenterCropSquare:
            return "crop";
        case PreprocessMode::ResizeStretch:
            return "stretch";
        case PreprocessMode::LetterboxSquare:
            return "letterbox";
    }
    return "crop";
}

inline PreprocessMode preprocess_mode_from_string(const std::string& s) {
    if (s == "crop" || s == "center-crop" || s == "centercrop") {
        return PreprocessMode::CenterCropSquare;
    }
    if (s == "stretch" || s == "resize") {
        return PreprocessMode::ResizeStretch;
    }
    if (s == "letterbox" || s == "pad") {
        return PreprocessMode::LetterboxSquare;
    }
    throw std::runtime_error("Unknown preprocess mode: " + s + ". Use crop|stretch|letterbox.");
}

inline std::array<float, 3> parse_triplet_csv(const std::string& csv) {
    std::array<float, 3> out{};
    std::stringstream ss(csv);
    std::string token;
    for (int i = 0; i < 3; ++i) {
        if (!std::getline(ss, token, ',')) {
            throw std::runtime_error("Expected 3 comma-separated floats, got: " + csv);
        }
        out[i] = std::stof(token);
    }
    if (std::getline(ss, token, ',')) {
        throw std::runtime_error("Expected exactly 3 comma-separated floats, got: " + csv);
    }
    return out;
}

inline bool is_image_file(const fs::path& p) {
    if (!fs::is_regular_file(p)) {
        return false;
    }
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    static const std::vector<std::string> kExts = {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
    };
    return std::find(kExts.begin(), kExts.end(), ext) != kExts.end();
}

inline std::vector<fs::path> collect_images(const fs::path& input, bool recursive) {
    std::vector<fs::path> out;
    if (fs::is_regular_file(input)) {
        if (is_image_file(input)) {
            out.push_back(input);
        }
        return out;
    }
    if (!fs::is_directory(input)) {
        throw std::runtime_error("Input path does not exist or is not a file/directory: " + input.string());
    }

    if (recursive) {
        for (const auto& entry : fs::recursive_directory_iterator(input)) {
            if (is_image_file(entry.path())) {
                out.push_back(entry.path());
            }
        }
    } else {
        for (const auto& entry : fs::directory_iterator(input)) {
            if (is_image_file(entry.path())) {
                out.push_back(entry.path());
            }
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace portable_cls
