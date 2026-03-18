
#pragma once

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <ncnn/net.h>

#include "classifier_common.hpp"

namespace vision_app {

class NcnnClassifier {
public:
    struct Config {
        int input_width = 128;
        int input_height = 128;
        std::string preprocess = "crop";
        int topk = 5;
        int threads = 4;
    };

    bool load(const std::string& param_path,
              const std::string& bin_path,
              const std::string& labels_path,
              const Config& cfg,
              std::string& err) {
        cfg_ = cfg;
        if (!load_labels_txt(labels_path, labels_, err)) return false;
        net_.opt.use_vulkan_compute = false;
        net_.opt.num_threads = std::max(1, cfg_.threads);
        if (net_.load_param(param_path.c_str()) != 0) { err = "failed to load param: " + param_path; return false; }
        if (net_.load_model(bin_path.c_str()) != 0) { err = "failed to load bin: " + bin_path; return false; }
        if (!infer_io_names_from_param(param_path, input_blob_name_, output_blob_name_)) {
            input_blob_name_ = "in0";
            output_blob_name_ = "out0";
        }
        err.clear();
        return true;
    }

    ClassifyResult classify(const cv::Mat& image_bgr) const {
        ClassifyResult r;
        if (image_bgr.empty()) { r.summary = "empty roi"; return r; }
        cv::Mat prep = preprocess_bgr(image_bgr, cfg_.input_width, cfg_.input_height, cfg_.preprocess);
        if (prep.empty()) { r.summary = "prep failed"; return r; }

        const auto t0 = std::chrono::steady_clock::now();
        ncnn::Mat in = ncnn::Mat::from_pixels(prep.data, ncnn::Mat::PIXEL_BGR2RGB, prep.cols, prep.rows);
        const float norm_vals[3] = {1.f/255.f, 1.f/255.f, 1.f/255.f};
        in.substract_mean_normalize(nullptr, norm_vals);
        ncnn::Extractor ex = net_.create_extractor();
        if (ex.input(input_blob_name_.c_str(), in) != 0) {
            r.summary = "ncnn input failed";
            return r;
        }
        ncnn::Mat out;
        if (ex.extract(output_blob_name_.c_str(), out) != 0) {
            r.summary = "ncnn extract failed";
            return r;
        }
        const auto t1 = std::chrono::steady_clock::now();
        r.infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::vector<float> scores(out.w);
        for (int i = 0; i < out.w; ++i) scores[i] = out[i];
        auto idx = topk_indices(scores, cfg_.topk);
        r.ok = true;
        for (int i : idx) {
            ClassHit h;
            h.index = i;
            h.score = scores[i];
            h.label = (i >= 0 && i < static_cast<int>(labels_.size())) ? labels_[i] : ("cls_" + std::to_string(i));
            r.topk.push_back(h);
        }
        if (!r.topk.empty()) r.best = r.topk.front();
        r.summary = make_summary(r);
        return r;
    }

private:
    static bool infer_io_names_from_param(const std::string& param_path, std::string& in_name, std::string& out_name) {
        std::ifstream in(param_path);
        if (!in.is_open()) return false;
        std::string line;
        std::vector<std::string> lines;
        while (std::getline(in, line)) if (!line.empty()) lines.push_back(line);
        if (lines.size() < 3) return false;
        auto split_ws = [](const std::string& s) {
            std::istringstream iss(s);
            std::vector<std::string> tok; std::string t;
            while (iss >> t) tok.push_back(t);
            return tok;
        };
        auto first = split_ws(lines[2]);
        auto last = split_ws(lines.back());
        if (first.size() < 5 || last.size() < 5) return false;
        in_name = first[4];
        out_name = last.back();
        return true;
    }

    Config cfg_;
    std::vector<std::string> labels_;
    mutable ncnn::Net net_;
    std::string input_blob_name_;
    std::string output_blob_name_;
};

} // namespace vision_app
