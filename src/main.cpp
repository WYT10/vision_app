#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "app_config.hpp"
#include "deploy_runtime.hpp"

using namespace vision_app;

namespace {

std::string need(int& i, int argc, char** argv, const char* flag) {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
    return argv[++i];
}

bool parse_bool(const std::string& v) {
    return v == "1" || v == "true" || v == "TRUE" || v == "yes" || v == "on";
}

bool parse_roi_csv(const std::string& s, RoiRatio& r) {
    std::stringstream ss(s);
    std::string tok;
    double vals[4];
    int j = 0;
    while (std::getline(ss, tok, ',') && j < 4) vals[j++] = std::stod(tok);
    if (j != 4) return false;
    r.x = vals[0]; r.y = vals[1]; r.w = vals[2]; r.h = vals[3];
    r = clamp_roi(r);
    return true;
}

bool parse_band_csv(const std::string& s, BandRatio& b) {
    std::stringstream ss(s);
    std::string tok;
    double vals[2];
    int j = 0;
    while (std::getline(ss, tok, ',') && j < 2) vals[j++] = std::stod(tok);
    if (j != 2) return false;
    b.y = std::clamp(vals[0], 0.0, 0.98);
    b.h = std::clamp(vals[1], 0.01, 1.0 - b.y);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    AppOptions opt;
    std::string config_path = opt.profile_path;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--help" || a == "-h") { print_help(); return 0; }
            if (a == "--config") config_path = need(i, argc, argv, "--config");
        }

        opt.profile_path = config_path;
        std::string load_err;
        load_profile_config(config_path, opt, load_err);

        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--help" || a == "-h") {}
            else if (a == "--config") { ++i; }
            else if (a == "--mode") opt.mode = need(i, argc, argv, "--mode");
            else if (a == "--device") opt.device = need(i, argc, argv, "--device");
            else if (a == "--width") opt.width = std::stoi(need(i, argc, argv, "--width"));
            else if (a == "--height") opt.height = std::stoi(need(i, argc, argv, "--height"));
            else if (a == "--fps") opt.fps = std::stoi(need(i, argc, argv, "--fps"));
            else if (a == "--fourcc") opt.fourcc = need(i, argc, argv, "--fourcc");
            else if (a == "--buffer-size") opt.buffer_size = std::stoi(need(i, argc, argv, "--buffer-size"));
            else if (a == "--latest-only") opt.latest_only = parse_bool(need(i, argc, argv, "--latest-only"));
            else if (a == "--drain-grabs") opt.drain_grabs = std::stoi(need(i, argc, argv, "--drain-grabs"));
            else if (a == "--ui") opt.ui = parse_bool(need(i, argc, argv, "--ui"));
            else if (a == "--mobile-webcam") opt.mobile_webcam = parse_bool(need(i, argc, argv, "--mobile-webcam"));
            else if (a == "--draw-overlay") opt.draw_overlay = parse_bool(need(i, argc, argv, "--draw-overlay"));
            else if (a == "--duration") opt.duration = std::stoi(need(i, argc, argv, "--duration"));
            else if (a == "--camera-preview-max") opt.camera_preview_max = std::stoi(need(i, argc, argv, "--camera-preview-max"));
            else if (a == "--warp-preview-max") opt.warp_preview_max = std::stoi(need(i, argc, argv, "--warp-preview-max"));
            else if (a == "--status-width") opt.status_width = std::stoi(need(i, argc, argv, "--status-width"));
            else if (a == "--show-status-window") opt.show_status_window = parse_bool(need(i, argc, argv, "--show-status-window"));
            else if (a == "--text-sink") opt.text_sink = need(i, argc, argv, "--text-sink");
            else if (a == "--warp-width") opt.warp_width = std::stoi(need(i, argc, argv, "--warp-width"));
            else if (a == "--warp-height") opt.warp_height = std::stoi(need(i, argc, argv, "--warp-height"));
            else if (a == "--target-tag-px") opt.target_tag_px = std::stoi(need(i, argc, argv, "--target-tag-px"));
            else if (a == "--tag-family") opt.tag_family = need(i, argc, argv, "--tag-family");
            else if (a == "--target-id") opt.target_id = std::stoi(need(i, argc, argv, "--target-id"));
            else if (a == "--require-target-id") opt.require_target_id = parse_bool(need(i, argc, argv, "--require-target-id"));
            else if (a == "--manual-lock-only") opt.manual_lock_only = parse_bool(need(i, argc, argv, "--manual-lock-only"));
            else if (a == "--lock-frames") opt.lock_frames = std::stoi(need(i, argc, argv, "--lock-frames"));
            else if (a == "--save-warp") opt.save_warp = need(i, argc, argv, "--save-warp");
            else if (a == "--load-warp") opt.load_warp = need(i, argc, argv, "--load-warp");
            else if (a == "--save-rois") opt.save_rois = need(i, argc, argv, "--save-rois");
            else if (a == "--load-rois") opt.load_rois = need(i, argc, argv, "--load-rois");
            else if (a == "--save-report") opt.save_report = need(i, argc, argv, "--save-report");
            else if (a == "--trigger-mode") opt.trigger_mode = need(i, argc, argv, "--trigger-mode");
            else if (a == "--red-roi") { if (!parse_roi_csv(need(i, argc, argv, "--red-roi"), opt.default_rois.red_roi)) throw std::runtime_error("bad --red-roi"); }
            else if (a == "--image-roi") { if (!parse_roi_csv(need(i, argc, argv, "--image-roi"), opt.default_rois.image_roi)) throw std::runtime_error("bad --image-roi"); }
            else if (a == "--upper-band") { if (!parse_band_csv(need(i, argc, argv, "--upper-band"), opt.dynamic_cfg.upper_band)) throw std::runtime_error("bad --upper-band"); }
            else if (a == "--lower-band") { if (!parse_band_csv(need(i, argc, argv, "--lower-band"), opt.dynamic_cfg.lower_band)) throw std::runtime_error("bad --lower-band"); }
            else if (a == "--image-bottom-offset") opt.dynamic_cfg.image_roi.bottom_offset = std::stod(need(i, argc, argv, "--image-bottom-offset"));
            else if (a == "--dynamic-image-width") opt.dynamic_cfg.image_roi.width = std::stod(need(i, argc, argv, "--dynamic-image-width"));
            else if (a == "--dynamic-image-height") opt.dynamic_cfg.image_roi.height = std::stod(need(i, argc, argv, "--dynamic-image-height"));
            else if (a == "--red-ratio-threshold") opt.fixed_cfg.red_ratio_threshold = std::stod(need(i, argc, argv, "--red-ratio-threshold"));
            else if (a == "--min-red-width-ratio") opt.dynamic_cfg.min_red_width_ratio = std::stod(need(i, argc, argv, "--min-red-width-ratio"));
            else if (a == "--min-red-fill-ratio") opt.dynamic_cfg.min_red_fill_ratio = std::stod(need(i, argc, argv, "--min-red-fill-ratio"));
            else if (a == "--red-h1-low") opt.red_cfg.h1_low = std::stoi(need(i, argc, argv, "--red-h1-low"));
            else if (a == "--red-h1-high") opt.red_cfg.h1_high = std::stoi(need(i, argc, argv, "--red-h1-high"));
            else if (a == "--red-h2-low") opt.red_cfg.h2_low = std::stoi(need(i, argc, argv, "--red-h2-low"));
            else if (a == "--red-h2-high") opt.red_cfg.h2_high = std::stoi(need(i, argc, argv, "--red-h2-high"));
            else if (a == "--red-s-min") opt.red_cfg.s_min = std::stoi(need(i, argc, argv, "--red-s-min"));
            else if (a == "--red-v-min") opt.red_cfg.v_min = std::stoi(need(i, argc, argv, "--red-v-min"));
            else if (a == "--model-enable") opt.model_cfg.enable = parse_bool(need(i, argc, argv, "--model-enable"));
            else if (a == "--model-backend") opt.model_cfg.backend = need(i, argc, argv, "--model-backend");
            else if (a == "--model-onnx-path") opt.model_cfg.onnx_path = need(i, argc, argv, "--model-onnx-path");
            else if (a == "--model-ncnn-param-path") opt.model_cfg.ncnn_param_path = need(i, argc, argv, "--model-ncnn-param-path");
            else if (a == "--model-ncnn-bin-path") opt.model_cfg.ncnn_bin_path = need(i, argc, argv, "--model-ncnn-bin-path");
            else if (a == "--model-labels-path") opt.model_cfg.labels_path = need(i, argc, argv, "--model-labels-path");
            else if (a == "--model-input-width") opt.model_cfg.input_width = std::stoi(need(i, argc, argv, "--model-input-width"));
            else if (a == "--model-input-height") opt.model_cfg.input_height = std::stoi(need(i, argc, argv, "--model-input-height"));
            else if (a == "--model-preprocess") opt.model_cfg.preprocess = need(i, argc, argv, "--model-preprocess");
            else if (a == "--model-threads") opt.model_cfg.threads = std::stoi(need(i, argc, argv, "--model-threads"));
            else if (a == "--model-stride") opt.model_cfg.stride = std::stoi(need(i, argc, argv, "--model-stride"));
            else if (a == "--model-topk") opt.model_cfg.topk = std::stoi(need(i, argc, argv, "--model-topk"));
            else if (a == "--model-max-hz") opt.model_max_hz = std::stod(need(i, argc, argv, "--model-max-hz"));
            else if (a == "--run-red") opt.run_red = parse_bool(need(i, argc, argv, "--run-red"));
            else if (a == "--run-image-roi") opt.run_image_roi = parse_bool(need(i, argc, argv, "--run-image-roi"));
            else if (a == "--run-model") opt.run_model = parse_bool(need(i, argc, argv, "--run-model"));
            else if (a == "--save-image-roi-dir") opt.save_image_roi_dir = need(i, argc, argv, "--save-image-roi-dir");
            else if (a == "--save-red-roi-dir") opt.save_red_roi_dir = need(i, argc, argv, "--save-red-roi-dir");
            else if (a == "--save-every-n") opt.save_every_n = std::stoi(need(i, argc, argv, "--save-every-n"));
            else throw std::runtime_error("unknown argument: " + a);
        }

        if (opt.mobile_webcam) {
            std::cout << "[INFO] Mobile webcam mode enabled: throttling to ~10fps, setting arbitrary backend\n";
            opt.drain_grabs = 2; // skip 2 frames per read, 30fps -> 10fps
            opt.fourcc = ""; // Let backend decide safely
            opt.buffer_size = 1;
            opt.latest_only = true;
        }

        std::string err;
        if (opt.mode == "probe") return run_probe(opt, err) ? 0 : (std::cerr << "Probe failed: " << err << "\n", 1);
        if (opt.mode == "live") return run_live(opt, err) ? 0 : (std::cerr << "Live failed: " << err << "\n", 1);
        if (opt.mode == "calibrate") return run_calibrate(opt, err) ? 0 : (std::cerr << "Calibrate failed: " << err << "\n", 1);
        if (opt.mode == "deploy") return run_deploy(opt, err) ? 0 : (std::cerr << "Deploy failed: " << err << "\n", 1);

        std::cerr << "Unknown mode: " << opt.mode << "\n";
        print_help();
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
