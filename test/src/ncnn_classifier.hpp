#pragma once

#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <net.h>
#include <opencv2/core.hpp>

#include "classifier_common.hpp"

namespace portable_cls {

class NcnnClassifier {
public:
    struct Config : public CommonConfig {
        bool use_vulkan = false;
        bool auto_detect_blob_names = true;
        std::string input_blob;
        std::string output_blob;
    };

    bool load(const fs::path& param_path,
              const fs::path& bin_path,
              const fs::path& labels_path,
              const Config& cfg = Config()) {
        cfg_ = cfg;
        labels_ = read_labels(labels_path);

        if (cfg_.auto_detect_blob_names) {
            const auto blobs = infer_blob_names_from_param(param_path);
            if (cfg_.input_blob.empty()) cfg_.input_blob = blobs.first;
            if (cfg_.output_blob.empty()) cfg_.output_blob = blobs.second;
        }

        if (cfg_.input_blob.empty() || cfg_.output_blob.empty()) {
            throw std::runtime_error("NCNN input/output blob names are empty. Set them manually or enable auto detection.");
        }

        net_.clear();
        net_.opt.use_vulkan_compute = cfg_.use_vulkan;
        net_.opt.use_fp16_packed = true;
        net_.opt.use_fp16_storage = true;
        net_.opt.use_fp16_arithmetic = false;  // safer default on CPU-only boards

        if (net_.load_param(param_path.string().c_str()) != 0) {
            throw std::runtime_error("Failed to load ncnn param: " + param_path.string());
        }
        if (net_.load_model(bin_path.string().c_str()) != 0) {
            throw std::runtime_error("Failed to load ncnn bin: " + bin_path.string());
        }
        return true;
    }

    Result classify(const cv::Mat& image_bgr, int topk_override = -1) const {
        if (image_bgr.empty()) {
            throw std::runtime_error("NCNN classify() received empty image");
        }
        const cv::Mat prepared = preprocess_image(image_bgr, cfg_);

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(
            prepared.data,
            ncnn::Mat::PIXEL_BGR2RGB,
            prepared.cols,
            prepared.rows,
            cfg_.input_width,
            cfg_.input_height);
        in.substract_mean_normalize(cfg_.mean_vals.data(), cfg_.norm_vals.data());

        ncnn::Extractor ex = net_.create_extractor();
        ex.set_light_mode(true);
        ex.set_num_threads(cfg_.num_threads);

        if (ex.input(cfg_.input_blob.c_str(), in) != 0) {
            throw std::runtime_error("NCNN input failed for blob: " + cfg_.input_blob);
        }

        ncnn::Mat out;
        if (ex.extract(cfg_.output_blob.c_str(), out) != 0) {
            throw std::runtime_error("NCNN extract failed for blob: " + cfg_.output_blob);
        }

        const std::vector<float> raw = flatten_to_vector(out);
        if (raw.empty()) {
            throw std::runtime_error("NCNN output is empty");
        }

        Result r;
        r.probabilities = probabilities_from_output(raw);
        const int use_topk = (topk_override > 0) ? topk_override : cfg_.topk;
        r.topk = select_topk(r.probabilities, labels_, use_topk);
        if (!r.topk.empty()) {
            r.best_index = r.topk.front().index;
            r.best_probability = r.topk.front().probability;
            r.best_label = r.topk.front().label;
        }
        return r;
    }

    const Config& config() const { return cfg_; }
    const std::vector<std::string>& labels() const { return labels_; }

private:
    ncnn::Net net_;
    Config cfg_{};
    std::vector<std::string> labels_;

    static std::vector<float> flatten_to_vector(const ncnn::Mat& out) {
        std::vector<float> values;
        const int total = out.w * out.h * out.d * out.c;
        values.resize(total);
        if (total == 0) {
            return values;
        }
        const float* ptr = static_cast<const float*>(out.data);
        std::copy(ptr, ptr + total, values.begin());
        return values;
    }

    static std::pair<std::string, std::string> infer_blob_names_from_param(const fs::path& param_path) {
        std::ifstream ifs(param_path);
        if (!ifs) {
            throw std::runtime_error("Failed to open param file for blob-name inference: " + param_path.string());
        }

        std::string input_blob;
        std::vector<std::string> all_tops;
        std::unordered_set<std::string> all_bottoms;
        std::string line;

        while (std::getline(ifs, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }

            std::istringstream iss(line);
            std::vector<std::string> tok;
            for (std::string t; iss >> t;) {
                tok.push_back(t);
            }
            if (tok.size() < 4) {
                continue;
            }

            int bottom_count = 0;
            int top_count = 0;
            try {
                bottom_count = std::stoi(tok[2]);
                top_count = std::stoi(tok[3]);
            } catch (...) {
                continue;
            }

            const int blobs_start = 4;
            if (static_cast<int>(tok.size()) < blobs_start + bottom_count + top_count) {
                continue;
            }

            const std::string& layer_type = tok[0];
            const auto bottoms_begin = tok.begin() + blobs_start;
            const auto tops_begin = bottoms_begin + bottom_count;

            for (int i = 0; i < bottom_count; ++i) {
                all_bottoms.insert(*(bottoms_begin + i));
            }
            for (int i = 0; i < top_count; ++i) {
                all_tops.push_back(*(tops_begin + i));
            }

            if (layer_type == "Input" && top_count >= 1 && input_blob.empty()) {
                input_blob = *(tops_begin + 0);
            }
        }

        if (input_blob.empty()) {
            throw std::runtime_error("Failed to infer NCNN input blob from param file: " + param_path.string());
        }

        std::string output_blob;
        for (auto it = all_tops.rbegin(); it != all_tops.rend(); ++it) {
            if (!all_bottoms.count(*it)) {
                output_blob = *it;
                break;
            }
        }
        if (output_blob.empty() && !all_tops.empty()) {
            output_blob = all_tops.back();
        }
        if (output_blob.empty()) {
            throw std::runtime_error("Failed to infer NCNN output blob from param file: " + param_path.string());
        }

        return {input_blob, output_blob};
    }
};

}  // namespace portable_cls
