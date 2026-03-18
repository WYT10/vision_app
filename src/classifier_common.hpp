
#pragma once

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace vision_app {

struct ClassHit {
    int index = -1;
    float score = 0.f;
    std::string label;
};

struct ClassifyResult {
    bool ok = false;
    ClassHit best;
    std::vector<ClassHit> topk;
    double infer_ms = 0.0;
    std::string summary;
};

inline bool load_labels_txt(const std::string& path, std::vector<std::string>& labels, std::string& err) {
    labels.clear();
    std::ifstream in(path);
    if (!in.is_open()) { err = "cannot open labels: " + path; return false; }
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) labels.push_back(line);
    }
    if (labels.empty()) { err = "labels file empty: " + path; return false; }
    err.clear();
    return true;
}

inline cv::Mat center_crop_square(const cv::Mat& img) {
    if (img.empty()) return img;
    const int side = std::min(img.cols, img.rows);
    const int x = std::max(0, (img.cols - side) / 2);
    const int y = std::max(0, (img.rows - side) / 2);
    return img(cv::Rect(x, y, side, side)).clone();
}

inline cv::Mat preprocess_bgr(const cv::Mat& bgr,
                              int out_w,
                              int out_h,
                              const std::string& mode) {
    if (bgr.empty()) return {};
    cv::Mat work;
    if (mode == "stretch") work = bgr;
    else work = center_crop_square(bgr);
    cv::Mat out;
    cv::resize(work, out, cv::Size(out_w, out_h), 0, 0, cv::INTER_LINEAR);
    return out;
}

inline std::string short_float(float v, int n = 4) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(n);
    oss << v;
    return oss.str();
}

inline std::vector<int> topk_indices(const std::vector<float>& scores, int k) {
    std::vector<int> idx(scores.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
    std::partial_sort(idx.begin(), idx.begin() + std::min<int>(k, idx.size()), idx.end(),
        [&](int a, int b) { return scores[a] > scores[b]; });
    if (static_cast<int>(idx.size()) > k) idx.resize(k);
    return idx;
}

inline std::string make_summary(const ClassifyResult& r, int topk_show = 5) {
    if (!r.ok) return "model err";
    std::ostringstream oss;
    oss << "best: [" << r.best.index << "] " << r.best.label << " prob=" << short_float(r.best.score, 4);
    if (!r.topk.empty()) {
        oss << " | top" << std::min<int>(topk_show, r.topk.size()) << ":";
        for (int i = 0; i < std::min<int>(topk_show, r.topk.size()); ++i) {
            const auto& h = r.topk[i];
            oss << " [" << h.index << "]" << h.label << '=' << short_float(h.score, 4);
        }
    }
    return oss.str();
}

} // namespace vision_app
