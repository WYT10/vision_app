#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "classifier_common.hpp"

#if defined(HAVE_ONNX_RUNTIME)
#include "onnx_classifier.hpp"
#endif
#if defined(HAVE_NCNN)
#include "ncnn_classifier.hpp"
#endif

namespace fs = std::filesystem;
using portable_cls::CommonConfig;
using portable_cls::PreprocessMode;
using portable_cls::Result;

struct Options {
    std::string backend;
    fs::path model_path;
    fs::path weights_path;  // ncnn .bin only
    fs::path labels_path;
    fs::path input_path;
    int size = 224;
    int topk = 5;
    int threads = 4;
    bool recursive = true;
    bool eval_parent_label = false;
    PreprocessMode preprocess = PreprocessMode::CenterCropSquare;
    std::array<float, 3> mean_vals{0.f, 0.f, 0.f};
    std::array<float, 3> norm_vals{1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
};

static void print_usage(const char* argv0) {
    std::cout << "Portable classifier inference\n\n"
              << "Usage:\n"
              << "  " << argv0 << " --backend onnx --model best.onnx --labels labels.txt --input image.jpg [options]\n"
              << "  " << argv0 << " --backend ncnn --model best.param --weights best.bin --labels labels.txt --input image.jpg [options]\n\n"
              << "Options:\n"
              << "  --backend onnx|ncnn        Backend to run\n"
              << "  --model PATH               Model path (.onnx or .param)\n"
              << "  --weights PATH             NCNN weights path (.bin)\n"
              << "  --labels PATH              labels.txt exported from training dataset\n"
              << "  --input PATH               Image file or directory\n"
              << "  --size N                   Input size (default: 224)\n"
              << "  --threads N                Inference threads (default: 4)\n"
              << "  --topk N                   Top-k predictions to print (default: 5)\n"
              << "  --prep crop|stretch|letterbox   Preprocess mode (default: crop)\n"
              << "  --mean a,b,c               Mean values (default: 0,0,0)\n"
              << "  --norm a,b,c               Normalization multipliers (default: 1/255 each)\n"
              << "  --eval-parent-label        Treat parent folder name as ground-truth label\n"
              << "  --no-recursive             Do not recurse into subdirectories\n"
              << "  --help                     Show this help\n\n"
              << "Examples:\n"
              << "  " << argv0 << " --backend onnx --model best.onnx --labels labels.txt --input test.jpg --prep crop\n"
              << "  " << argv0 << " --backend ncnn --model best.param --weights best.bin --labels labels.txt --input ./dataset_cls/test --eval-parent-label\n";
}

static Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto require_value = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + a);
            }
            return argv[++i];
        };

        if (a == "--backend") o.backend = require_value();
        else if (a == "--model") o.model_path = require_value();
        else if (a == "--weights") o.weights_path = require_value();
        else if (a == "--labels") o.labels_path = require_value();
        else if (a == "--input") o.input_path = require_value();
        else if (a == "--size") o.size = std::stoi(require_value());
        else if (a == "--threads") o.threads = std::stoi(require_value());
        else if (a == "--topk") o.topk = std::stoi(require_value());
        else if (a == "--prep") o.preprocess = portable_cls::preprocess_mode_from_string(require_value());
        else if (a == "--mean") o.mean_vals = portable_cls::parse_triplet_csv(require_value());
        else if (a == "--norm") o.norm_vals = portable_cls::parse_triplet_csv(require_value());
        else if (a == "--eval-parent-label") o.eval_parent_label = true;
        else if (a == "--no-recursive") o.recursive = false;
        else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + a);
        }
    }

    if (o.backend.empty()) throw std::runtime_error("--backend is required");
    if (o.model_path.empty()) throw std::runtime_error("--model is required");
    if (o.labels_path.empty()) throw std::runtime_error("--labels is required");
    if (o.input_path.empty()) throw std::runtime_error("--input is required");
    if (o.backend == "ncnn" && o.weights_path.empty()) {
        throw std::runtime_error("--weights is required for NCNN backend");
    }
    return o;
}

class IClassifier {
public:
    virtual ~IClassifier() = default;
    virtual Result classify(const cv::Mat& image, int topk) const = 0;
};

#if defined(HAVE_ONNX_RUNTIME)
class OnnxAdapter final : public IClassifier {
public:
    explicit OnnxAdapter(portable_cls::OnnxClassifier impl) : impl_(std::move(impl)) {}
    Result classify(const cv::Mat& image, int topk) const override { return impl_.classify(image, topk); }
private:
    portable_cls::OnnxClassifier impl_;
};
#endif

#if defined(HAVE_NCNN)
class NcnnAdapter final : public IClassifier {
public:
    explicit NcnnAdapter(portable_cls::NcnnClassifier impl) : impl_(std::move(impl)) {}
    Result classify(const cv::Mat& image, int topk) const override { return impl_.classify(image, topk); }
private:
    portable_cls::NcnnClassifier impl_;
};
#endif

static CommonConfig build_common_config(const Options& o) {
    CommonConfig cfg;
    cfg.input_width = o.size;
    cfg.input_height = o.size;
    cfg.topk = o.topk;
    cfg.num_threads = o.threads;
    cfg.preprocess = o.preprocess;
    cfg.mean_vals = o.mean_vals;
    cfg.norm_vals = o.norm_vals;
    return cfg;
}

static std::unique_ptr<IClassifier> make_classifier(const Options& o) {
    if (o.backend == "onnx") {
#if defined(HAVE_ONNX_RUNTIME)
        portable_cls::OnnxClassifier::Config cfg;
        static_cast<CommonConfig&>(cfg) = build_common_config(o);
        portable_cls::OnnxClassifier clf;
        clf.load(o.model_path, o.labels_path, cfg);
        return std::make_unique<OnnxAdapter>(std::move(clf));
#else
        throw std::runtime_error("This build does not include ONNX Runtime support. Reconfigure with -DENABLE_ONNX_RUNTIME=ON.");
#endif
    }
    if (o.backend == "ncnn") {
#if defined(HAVE_NCNN)
        portable_cls::NcnnClassifier::Config cfg;
        static_cast<CommonConfig&>(cfg) = build_common_config(o);
        cfg.use_vulkan = false;
        portable_cls::NcnnClassifier clf;
        clf.load(o.model_path, o.weights_path, o.labels_path, cfg);
        return std::make_unique<NcnnAdapter>(std::move(clf));
#else
        throw std::runtime_error("This build does not include NCNN support. Reconfigure with -DENABLE_NCNN=ON.");
#endif
    }
    throw std::runtime_error("Unsupported backend: " + o.backend + ". Use onnx or ncnn.");
}

static void print_result(const fs::path& path, const Result& r) {
    std::cout << path.string() << '\n';
    std::cout << "  best: [" << r.best_index << "] " << r.best_label
              << "  prob=" << std::fixed << std::setprecision(4) << r.best_probability << '\n';
    std::cout << "  top" << r.topk.size() << ':';
    for (const auto& s : r.topk) {
        std::cout << " [" << s.index << "]" << s.label << '=' << std::fixed << std::setprecision(4) << s.probability;
    }
    std::cout << "\n";
}

int main(int argc, char** argv) {
    try {
        const Options opt = parse_args(argc, argv);
        const auto classifier = make_classifier(opt);
        const std::vector<fs::path> images = portable_cls::collect_images(opt.input_path, opt.recursive);
        if (images.empty()) {
            throw std::runtime_error("No images found under: " + opt.input_path.string());
        }

        const auto t0 = std::chrono::steady_clock::now();
        int total = 0;
        int correct = 0;

        for (const auto& image_path : images) {
            cv::Mat img = cv::imread(image_path.string(), cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "Skipping unreadable image: " << image_path << '\n';
                continue;
            }
            const Result r = classifier->classify(img, opt.topk);
            print_result(image_path, r);
            ++total;

            if (opt.eval_parent_label) {
                const std::string expected = image_path.parent_path().filename().string();
                const bool ok = (r.best_label == expected);
                correct += ok ? 1 : 0;
                std::cout << "  expected: " << expected << "  match=" << (ok ? "yes" : "no") << "\n";
            }
        }

        const auto t1 = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "\nProcessed " << total << " image(s) in " << std::fixed << std::setprecision(3)
                  << seconds << " s";
        if (seconds > 0.0) {
            std::cout << "  (" << (static_cast<double>(total) / seconds) << " img/s)";
        }
        std::cout << '\n';

        if (opt.eval_parent_label && total > 0) {
            const double acc = static_cast<double>(correct) / total;
            std::cout << "Top-1 accuracy vs parent folder label: " << std::fixed << std::setprecision(4)
                      << acc << "  (" << correct << '/' << total << ")\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        print_usage(argv[0]);
        return 1;
    }
}
