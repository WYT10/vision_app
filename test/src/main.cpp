#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
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
    fs::path summary_json_path;
    fs::path per_class_csv_path;
    int size = 224;
    int topk = 5;
    int threads = 4;
    int warmup = 0;
    int repeat = 1;
    bool recursive = true;
    bool eval_parent_label = false;
    bool quiet_per_image = false;
    PreprocessMode preprocess = PreprocessMode::CenterCropSquare;
    std::array<float, 3> mean_vals{0.f, 0.f, 0.f};
    std::array<float, 3> norm_vals{1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
};

struct PerClassStat {
    int seen = 0;
    int correct = 0;
};

struct RunSummary {
    std::string backend;
    std::string input_path;
    std::string preprocess;
    int size = 224;
    int threads = 4;
    int topk = 5;
    int total = 0;
    int correct = 0;
    int source_images = 0;
    int repeat = 1;
    int warmup = 0;
    bool eval_parent_label = false;
    double seconds = 0.0;
    double img_per_s = 0.0;
    double ms_per_image = 0.0;
};

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    return out;
}

static void write_summary_json(const fs::path& path, const RunSummary& s) {
    std::ofstream ofs(path);
    if (!ofs) throw std::runtime_error("Failed to write summary JSON: " + path.string());
    const double acc = (s.total > 0) ? static_cast<double>(s.correct) / s.total : 0.0;
    ofs << "{\n";
    ofs << "  \"backend\": \"" << json_escape(s.backend) << "\",\n";
    ofs << "  \"input_path\": \"" << json_escape(s.input_path) << "\",\n";
    ofs << "  \"preprocess\": \"" << json_escape(s.preprocess) << "\",\n";
    ofs << "  \"size\": " << s.size << ",\n";
    ofs << "  \"threads\": " << s.threads << ",\n";
    ofs << "  \"topk\": " << s.topk << ",\n";
    ofs << "  \"source_images\": " << s.source_images << ",\n";
    ofs << "  \"repeat\": " << s.repeat << ",\n";
    ofs << "  \"warmup\": " << s.warmup << ",\n";
    ofs << "  \"eval_parent_label\": " << (s.eval_parent_label ? "true" : "false") << ",\n";
    ofs << "  \"total\": " << s.total << ",\n";
    ofs << "  \"correct\": " << s.correct << ",\n";
    ofs << std::fixed << std::setprecision(6);
    ofs << "  \"accuracy\": " << acc << ",\n";
    ofs << "  \"seconds\": " << s.seconds << ",\n";
    ofs << "  \"img_per_s\": " << s.img_per_s << ",\n";
    ofs << "  \"ms_per_image\": " << s.ms_per_image << "\n";
    ofs << "}\n";
}

static void write_per_class_csv(const fs::path& path, const std::map<std::string, PerClassStat>& stats) {
    std::ofstream ofs(path);
    if (!ofs) throw std::runtime_error("Failed to write per-class CSV: " + path.string());
    ofs << "label,seen,correct,accuracy\n";
    for (const auto& kv : stats) {
        const double acc = kv.second.seen > 0 ? static_cast<double>(kv.second.correct) / kv.second.seen : 0.0;
        ofs << kv.first << ',' << kv.second.seen << ',' << kv.second.correct << ','
            << std::fixed << std::setprecision(6) << acc << '\n';
    }
}

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
              << "  --summary-json PATH        Write run summary JSON\n"
              << "  --per-class-csv PATH       Write per-class accuracy CSV (requires --eval-parent-label)\n"
              << "  --quiet-per-image          Do not print each image prediction\n"
              << "  --repeat N                 Repeat a single input image N times for latency test\n"
              << "  --warmup N                 Warm up single-image inference N times before timing\n"
              << "  --no-recursive             Do not recurse into subdirectories\n"
              << "  --help                     Show this help\n\n"
              << "Examples:\n"
              << "  " << argv0 << " --backend onnx --model best.onnx --labels labels.txt --input test.jpg --prep crop\n"
              << "  " << argv0 << " --backend ncnn --model best.param --weights best.bin --labels labels.txt --input ./dataset_cls/test --eval-parent-label --quiet-per-image\n"
              << "  " << argv0 << " --backend ncnn --model best.param --weights best.bin --labels labels.txt --input test.jpg --repeat 200 --warmup 20 --quiet-per-image\n";
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
        else if (a == "--summary-json") o.summary_json_path = require_value();
        else if (a == "--per-class-csv") o.per_class_csv_path = require_value();
        else if (a == "--size") o.size = std::stoi(require_value());
        else if (a == "--threads") o.threads = std::stoi(require_value());
        else if (a == "--topk") o.topk = std::stoi(require_value());
        else if (a == "--warmup") o.warmup = std::stoi(require_value());
        else if (a == "--repeat") o.repeat = std::stoi(require_value());
        else if (a == "--prep") o.preprocess = portable_cls::preprocess_mode_from_string(require_value());
        else if (a == "--mean") o.mean_vals = portable_cls::parse_triplet_csv(require_value());
        else if (a == "--norm") o.norm_vals = portable_cls::parse_triplet_csv(require_value());
        else if (a == "--eval-parent-label") o.eval_parent_label = true;
        else if (a == "--quiet-per-image") o.quiet_per_image = true;
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
    if (o.repeat < 1) throw std::runtime_error("--repeat must be >= 1");
    if (o.warmup < 0) throw std::runtime_error("--warmup must be >= 0");
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
    explicit OnnxAdapter(std::unique_ptr<portable_cls::OnnxClassifier> impl) : impl_(std::move(impl)) {}
    Result classify(const cv::Mat& image, int topk) const override { return impl_->classify(image, topk); }
private:
    std::unique_ptr<portable_cls::OnnxClassifier> impl_;
};
#endif

#if defined(HAVE_NCNN)
class NcnnAdapter final : public IClassifier {
public:
    explicit NcnnAdapter(std::unique_ptr<portable_cls::NcnnClassifier> impl) : impl_(std::move(impl)) {}
    Result classify(const cv::Mat& image, int topk) const override { return impl_->classify(image, topk); }
private:
    std::unique_ptr<portable_cls::NcnnClassifier> impl_;
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
        auto clf = std::make_unique<portable_cls::OnnxClassifier>();
        clf->load(o.model_path, o.labels_path, cfg);
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
        auto clf = std::make_unique<portable_cls::NcnnClassifier>();
        clf->load(o.model_path, o.weights_path, o.labels_path, cfg);
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

static RunSummary print_and_store_summary(const Options& opt,
                                          int total,
                                          int correct,
                                          int source_images,
                                          double seconds) {
    RunSummary s;
    s.backend = opt.backend;
    s.input_path = opt.input_path.string();
    s.preprocess = portable_cls::preprocess_mode_to_string(opt.preprocess);
    s.size = opt.size;
    s.threads = opt.threads;
    s.topk = opt.topk;
    s.total = total;
    s.correct = correct;
    s.source_images = source_images;
    s.repeat = opt.repeat;
    s.warmup = opt.warmup;
    s.eval_parent_label = opt.eval_parent_label;
    s.seconds = seconds;
    s.img_per_s = seconds > 0.0 ? static_cast<double>(total) / seconds : 0.0;
    s.ms_per_image = total > 0 ? seconds * 1000.0 / static_cast<double>(total) : 0.0;

    std::cout << "\nProcessed " << total << " image(s) in " << std::fixed << std::setprecision(3)
              << seconds << " s";
    if (seconds > 0.0) {
        std::cout << "  (" << s.img_per_s << " img/s, " << s.ms_per_image << " ms/img)";
    }
    std::cout << '\n';

    if (opt.eval_parent_label && total > 0) {
        const double acc = static_cast<double>(correct) / total;
        std::cout << "Top-1 accuracy vs parent folder label: " << std::fixed << std::setprecision(4)
                  << acc << "  (" << correct << '/' << total << ")\n";
    }
    return s;
}

int main(int argc, char** argv) {
    try {
        const Options opt = parse_args(argc, argv);
        const auto classifier = make_classifier(opt);
        std::map<std::string, PerClassStat> per_class;

        int total = 0;
        int correct = 0;
        int source_images = 0;
        auto t0 = std::chrono::steady_clock::now();

        if (fs::is_regular_file(opt.input_path) && opt.repeat > 1) {
            cv::Mat img = cv::imread(opt.input_path.string(), cv::IMREAD_COLOR);
            if (img.empty()) throw std::runtime_error("Failed to read input image: " + opt.input_path.string());

            for (int i = 0; i < opt.warmup; ++i) {
                (void)classifier->classify(img, opt.topk);
            }
            t0 = std::chrono::steady_clock::now();
            Result last;
            for (int i = 0; i < opt.repeat; ++i) {
                last = classifier->classify(img, opt.topk);
                ++total;
            }
            const auto t1 = std::chrono::steady_clock::now();
            const double seconds = std::chrono::duration<double>(t1 - t0).count();
            source_images = 1;
            if (!opt.quiet_per_image) {
                print_result(opt.input_path, last);
                std::cout << "  repeated: " << opt.repeat << "  warmup: " << opt.warmup << '\n';
            }
            const RunSummary summary = print_and_store_summary(opt, total, correct, source_images, seconds);
            if (!opt.summary_json_path.empty()) write_summary_json(opt.summary_json_path, summary);
            return 0;
        }

        const std::vector<fs::path> images = portable_cls::collect_images(opt.input_path, opt.recursive);
        if (images.empty()) {
            throw std::runtime_error("No images found under: " + opt.input_path.string());
        }
        source_images = static_cast<int>(images.size());

        t0 = std::chrono::steady_clock::now();
        for (const auto& image_path : images) {
            cv::Mat img = cv::imread(image_path.string(), cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "Skipping unreadable image: " << image_path << '\n';
                continue;
            }
            const Result r = classifier->classify(img, opt.topk);
            ++total;

            if (!opt.quiet_per_image) {
                print_result(image_path, r);
            }

            if (opt.eval_parent_label) {
                const std::string expected = image_path.parent_path().filename().string();
                const bool ok = (r.best_label == expected);
                correct += ok ? 1 : 0;
                auto& stat = per_class[expected];
                stat.seen += 1;
                stat.correct += ok ? 1 : 0;
                if (!opt.quiet_per_image) {
                    std::cout << "  expected: " << expected << "  match=" << (ok ? "yes" : "no") << "\n";
                }
            }
        }

        const auto t1 = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        const RunSummary summary = print_and_store_summary(opt, total, correct, source_images, seconds);

        if (!opt.summary_json_path.empty()) write_summary_json(opt.summary_json_path, summary);
        if (!opt.per_class_csv_path.empty()) write_per_class_csv(opt.per_class_csv_path, per_class);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        print_usage(argv[0]);
        return 1;
    }
}
