#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <opencv2/imgcodecs.hpp>

#include "deploy.hpp"

using namespace vision_app;
namespace fs = std::filesystem;

static std::string trim(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

static std::string strip_inline_comment(const std::string& s) {
    const auto pos = s.find('#');
    if (pos == std::string::npos) return trim(s);
    return trim(s.substr(0, pos));
}

static std::string resolve_relative_path(const fs::path& base_dir, const std::string& raw) {
    if (raw.empty()) return raw;
    fs::path p(raw);
    if (p.is_absolute()) return p.string();
    return (base_dir / p).lexically_normal().string();
}

static void resolve_option_paths(const fs::path& config_path, AppOptions& o) {
    const fs::path base_dir = config_path.has_parent_path() ? config_path.parent_path() : fs::current_path();
    o.save_warp = resolve_relative_path(base_dir, o.save_warp);
    o.load_warp = resolve_relative_path(base_dir, o.load_warp);
    o.save_rois = resolve_relative_path(base_dir, o.save_rois);
    o.load_rois = resolve_relative_path(base_dir, o.load_rois);
    o.save_report = resolve_relative_path(base_dir, o.save_report);
    o.snap_path = resolve_relative_path(base_dir, o.snap_path);
    o.save_image_roi_dir = resolve_relative_path(base_dir, o.save_image_roi_dir);
    o.save_red_roi_dir = resolve_relative_path(base_dir, o.save_red_roi_dir);
    o.model_cfg.onnx_path = resolve_relative_path(base_dir, o.model_cfg.onnx_path);
    o.model_cfg.ncnn_param_path = resolve_relative_path(base_dir, o.model_cfg.ncnn_param_path);
    o.model_cfg.ncnn_bin_path = resolve_relative_path(base_dir, o.model_cfg.ncnn_bin_path);
    o.model_cfg.labels_path = resolve_relative_path(base_dir, o.model_cfg.labels_path);
}

static bool parse_bool(const std::string& v) {
    return v == "1" || v == "true" || v == "TRUE" || v == "yes" || v == "on";
}

static bool parse_roi_csv(const std::string& s, RoiRatio& r) {
    std::stringstream ss(s);
    std::string tok;
    double vals[4];
    int i = 0;
    while (std::getline(ss, tok, ',') && i < 4) vals[i++] = std::stod(trim(tok));
    if (i != 4) return false;
    r.x = vals[0];
    r.y = vals[1];
    r.w = vals[2];
    r.h = vals[3];
    r = clamp_roi(r);
    return true;
}

static void load_config(const std::string& path, AppOptions& o) {
    std::ifstream in(path);
    if (!in.is_open()) return;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        const auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        const std::string k = trim(line.substr(0, pos));
        const std::string v = strip_inline_comment(line.substr(pos + 1));

        if (k == "mode") o.mode = v;
        else if (k == "probe_task") o.probe_task = v;
        else if (k == "roi_mode") o.roi_mode = normalize_roi_mode(v);
        else if (k == "device") o.device = v;
        else if (k == "width") o.width = std::stoi(v);
        else if (k == "height") o.height = std::stoi(v);
        else if (k == "fps") o.fps = std::stoi(v);
        else if (k == "fourcc") o.fourcc = v;
        else if (k == "buffer_size") o.buffer_size = std::stoi(v);
        else if (k == "latest_only") o.latest_only = parse_bool(v);
        else if (k == "drain_grabs") o.drain_grabs = std::stoi(v);
        else if (k == "ui") o.ui = parse_bool(v);
        else if (k == "draw_overlay") o.draw_overlay = parse_bool(v);
        else if (k == "duration") o.duration = std::stoi(v);
        else if (k == "text_console") o.text_console = parse_bool(v);

        else if (k == "camera_soft_max") o.camera_soft_max = std::stoi(v);
        else if (k == "camera_preview_max") o.camera_preview_max = std::stoi(v);
        else if (k == "warp_preview_max") o.warp_preview_max = std::stoi(v);

        else if (k == "warp_width") o.warp_width = std::stoi(v);
        else if (k == "warp_height") o.warp_height = std::stoi(v);
        else if (k == "target_tag_px") o.target_tag_px = std::stoi(v);
        else if (k == "warp_center_x_ratio") o.warp_center_x_ratio = std::stod(v);
        else if (k == "warp_center_y_ratio") o.warp_center_y_ratio = std::stod(v);

        else if (k == "tag_family") o.tag_family = v;
        else if (k == "target_id") o.target_id = std::stoi(v);
        else if (k == "require_target_id") o.require_target_id = parse_bool(v);
        else if (k == "manual_lock_only") o.manual_lock_only = parse_bool(v);
        else if (k == "lock_frames") o.lock_frames = std::stoi(v);

        else if (k == "save_warp") o.save_warp = v;
        else if (k == "load_warp") o.load_warp = v;
        else if (k == "save_rois") o.save_rois = v;
        else if (k == "load_rois") o.load_rois = v;
        else if (k == "save_report") o.save_report = v;
        else if (k == "snap_path") o.snap_path = v;

        else if (k == "red_roi") parse_roi_csv(v, o.default_rois.red_roi);
        else if (k == "image_roi") parse_roi_csv(v, o.default_rois.image_roi);

        else if (k == "red_h1_low") o.red_cfg.h1_low = std::stoi(v);
        else if (k == "red_h1_high") o.red_cfg.h1_high = std::stoi(v);
        else if (k == "red_h2_low") o.red_cfg.h2_low = std::stoi(v);
        else if (k == "red_h2_high") o.red_cfg.h2_high = std::stoi(v);
        else if (k == "red_s_min") o.red_cfg.s_min = std::stoi(v);
        else if (k == "red_v_min") o.red_cfg.v_min = std::stoi(v);

        else if (k == "red_band_y0") o.dynamic_red.band_y0 = std::stoi(v);
        else if (k == "red_band_y1") o.dynamic_red.band_y1 = std::stoi(v);
        else if (k == "red_search_x0") o.dynamic_red.search_x0 = std::stoi(v);
        else if (k == "red_search_x1") o.dynamic_red.search_x1 = std::stoi(v);
        else if (k == "roi_gap_above_band") o.dynamic_red.roi_gap_above_band = std::stoi(v);
        else if (k == "roi_anchor_y") o.dynamic_red.roi_anchor_y = std::stoi(v);
        else if (k == "roi_width") o.dynamic_red.roi_width = std::stoi(v);
        else if (k == "roi_height") o.dynamic_red.roi_height = std::stoi(v);
        else if (k == "red_min_area") o.dynamic_red.min_area = std::stoi(v);
        else if (k == "red_max_area") o.dynamic_red.max_area = std::stoi(v);
        else if (k == "red_morph_k") o.dynamic_red.morph_k = std::stoi(v);
        else if (k == "red_miss_tolerance") o.dynamic_red.miss_tolerance = std::stoi(v);
        else if (k == "red_fallback_center_x") o.dynamic_red.fallback_center_x = std::stoi(v);
        else if (k == "red_center_alpha") o.dynamic_red.center_alpha = std::stod(v);
        else if (k == "red_show_mask_window") o.dynamic_red.show_mask_window = parse_bool(v);
        else if (k == "red_require_dual_zone") o.dynamic_red.require_dual_zone = parse_bool(v);
        else if (k == "red_zone_gap_px") o.dynamic_red.zone_gap_px = std::stoi(v);
        else if (k == "red_zone_min_pixels") o.dynamic_red.zone_min_pixels = std::stoi(v);
        else if (k == "red_zone_min_ratio") o.dynamic_red.zone_min_ratio = std::stod(v);
        else if (k == "red_band_min_pixels") o.dynamic_red.band_min_pixels = std::stoi(v);
        else if (k == "red_band_min_ratio") o.dynamic_red.band_min_ratio = std::stod(v);

        else if (k == "model_enable") o.model_cfg.enable = parse_bool(v);
        else if (k == "model_backend") o.model_cfg.backend = v;
        else if (k == "model_onnx_path") o.model_cfg.onnx_path = v;
        else if (k == "model_ncnn_param_path") o.model_cfg.ncnn_param_path = v;
        else if (k == "model_ncnn_bin_path") o.model_cfg.ncnn_bin_path = v;
        else if (k == "model_labels_path") o.model_cfg.labels_path = v;
        else if (k == "model_input_width") o.model_cfg.input_width = std::stoi(v);
        else if (k == "model_input_height") o.model_cfg.input_height = std::stoi(v);
        else if (k == "model_preprocess") o.model_cfg.preprocess = v;
        else if (k == "model_threads") o.model_cfg.threads = std::stoi(v);
        else if (k == "model_stride") o.model_cfg.stride = std::stoi(v);
        else if (k == "model_topk") o.model_cfg.topk = std::stoi(v);
        else if (k == "model_max_hz") o.model_max_hz = std::stod(v);

        else if (k == "run_red") o.run_red = parse_bool(v);
        else if (k == "run_image_roi") o.run_image_roi = parse_bool(v);
        else if (k == "run_model") o.run_model = parse_bool(v);
        else if (k == "save_image_roi_dir") o.save_image_roi_dir = v;
        else if (k == "save_red_roi_dir") o.save_red_roi_dir = v;
        else if (k == "save_every_n") o.save_every_n = std::stoi(v);
    }
}

static void print_help() {
    std::cout << R"TXT(
vision_app

Build:
  export CMAKE_PREFIX_PATH=/home/pi/ncnn/build/install:$CMAKE_PREFIX_PATH
  mkdir -p build
  cd build
  cmake ..
  make -j$(nproc)

Modes:
  --mode probe
  --mode calibrate
  --mode deploy

Probe tasks:
  --probe-task list|live|snap|bench
  --snap-path PATH

ROI runtime modes:
  --roi-mode fixed|dynamic-red-x

Core args:
  --config PATH
  --device /dev/video0
  --width 160 --height 120 --fourcc MJPG --fps 120
  --ui 1
  --headless
  --draw-overlay 1
  --duration 10
  --save-report ../report/latest_report.md

Calibration args:
  --warp-width 384 --warp-height 384
  --target-tag-px 128
  --warp-center-x-ratio 0.5 --warp-center-y-ratio 0.5
  --tag-family auto|16|25|36
  --target-id 0
  --manual-lock-only 1
  --save-warp ../report/warp_package.yml.gz
  --save-rois ../report/rois.yml

Fixed ROI args:
  --red-roi x,y,w,h
  --image-roi x,y,w,h

Dynamic red-x args:
  --red-band-y0 N --red-band-y1 N
  --red-search-x0 N --red-search-x1 N
  --roi-gap-above-band N
  --roi-anchor-y N   (legacy override)
  --roi-width N --roi-height N
  --red-min-area N --red-max-area N
  --red-morph-k N
  --red-center-alpha F
  --red-miss-tolerance N
  --red-fallback-center-x N
  --red-show-mask-window 1

Red threshold args:
  --red-h1-low N --red-h1-high N --red-h2-low N --red-h2-high N
  --red-s-min N --red-v-min N
  --red-require-dual-zone 1
  --red-zone-gap-px N
  --red-zone-min-pixels N
  --red-zone-min-ratio F
  --red-band-min-pixels N
  --red-band-min-ratio F

Model args:
  --model-enable 1
  --model-backend onnx|ncnn
  --model-onnx-path PATH
  --model-ncnn-param-path PATH
  --model-ncnn-bin-path PATH
  --model-labels-path PATH
  --model-input-width 128 --model-input-height 128
  --model-preprocess crop|stretch|letterbox
  --model-threads 4
  --model-stride 1
  --model-max-hz 5.0
  --model-topk 5

Partial pipeline / capture args:
  --run-red 1
  --run-image-roi 1
  --run-model 1
  --save-image-roi-dir PATH
  --save-red-roi-dir PATH
  --save-every-n 10

Examples:
  ./vision_app --mode probe --probe-task list --device /dev/video0
  ./vision_app --mode probe --probe-task bench --device /dev/video0 --headless --duration 10 --save-report ../report/probe_bench.md
  ./vision_app --config configs/pi5_fast.conf --mode calibrate --roi-mode fixed
  ./vision_app --config configs/pi5_fast.conf --mode calibrate --roi-mode dynamic-red-x --red-show-mask-window 1
  ./vision_app --config configs/pi5_fast.conf --mode deploy --roi-mode fixed
  ./vision_app --config configs/pi5_fast.conf --mode deploy --roi-mode dynamic-red-x
)TXT";
}

static bool run_probe_mode(const AppOptions& opt, std::string& err) {
    if (opt.probe_task == "list") {
        CameraProbeResult probe;
        if (!probe_camera(opt.device, probe, err)) return false;
        print_probe(probe);
        return true;
    }

    if (opt.probe_task == "snap") {
        int cam_w = opt.width;
        int cam_h = opt.height;
        cv::VideoCapture cap;
        if (!open_capture(cap, opt.device, cam_w, cam_h, opt.fps, opt.fourcc,
                          opt.buffer_size, opt.camera_soft_max, err)) {
            return false;
        }

        cv::Mat frame;
        for (int i = 0; i < 5; ++i) {
            if (!grab_latest_frame(cap, opt.latest_only, opt.drain_grabs, frame)) {
                err = "failed to warm up / read frame";
                return false;
            }
        }
        if (frame.empty()) {
            err = "empty frame during snap";
            return false;
        }

        const fs::path out_path = opt.snap_path.empty()
            ? fs::path("../captures/probe_snap.jpg")
            : fs::path(opt.snap_path);
        if (out_path.has_parent_path()) fs::create_directories(out_path.parent_path());
        if (!cv::imwrite(out_path.string(), frame)) {
            err = "failed to save image: " + out_path.string();
            return false;
        }

        std::cout << "Saved snap -> " << out_path << "\n";
        std::cout << "Frame size  -> " << frame.cols << 'x' << frame.rows << "\n";
        if (!opt.save_report.empty()) {
            write_report_md(opt.save_report,
                            "Probe Snap Report",
                            nullptr,
                            std::string("- Device: ") + opt.device + "\n" +
                            "- Saved image: " + out_path.string() + "\n" +
                            "- Frame size: " + std::to_string(frame.cols) + "x" + std::to_string(frame.rows) + "\n" +
                            "- Requested mode: " + std::to_string(opt.width) + "x" + std::to_string(opt.height) +
                            " @ " + std::to_string(opt.fps) + " fps" + "\n");
        }
        return true;
    }

    if (opt.probe_task == "live" || opt.probe_task == "bench") {
        RuntimeStats stats;
        const bool headless = !opt.ui;
        const int duration = std::max(1, opt.duration);
        if (!bench_capture(opt.device,
                           opt.width,
                           opt.height,
                           opt.fps,
                           opt.fourcc,
                           opt.buffer_size,
                           opt.latest_only,
                           opt.drain_grabs,
                           headless,
                           duration,
                           opt.camera_soft_max,
                           opt.camera_preview_max,
                           stats,
                           err)) {
            return false;
        }

        if (opt.probe_task == "bench") {
            print_runtime_stats(stats);
            if (!opt.save_report.empty()) {
                write_report_md(opt.save_report,
                                "Probe Bench Report",
                                &stats,
                                std::string("- Device: ") + opt.device + "\n" +
                                "- Requested mode: " + std::to_string(opt.width) + "x" + std::to_string(opt.height) +
                                " @ " + std::to_string(opt.fps) + " fps" + "\n" +
                                "- FOURCC: " + opt.fourcc + "\n" +
                                "- latest_only: " + std::string(opt.latest_only ? "true" : "false") + "\n" +
                                "- drain_grabs: " + std::to_string(opt.drain_grabs));
            }
        }
        return true;
    }

    err = "unknown --probe-task: " + opt.probe_task + ". Use list|live|snap|bench.";
    return false;
}

int main(int argc, char** argv) {
    AppOptions opt;
    opt.width = 160;
    opt.height = 120;
    opt.fps = 120;
    opt.fourcc = "MJPG";
    opt.model_max_hz = 5.0;
    std::string config_path = "vision_app.conf";

    auto need = [&](int& i, const char* flag) -> std::string {
        if (i + 1 >= argc) {
            throw std::runtime_error(std::string("missing value for ") + flag);
        }
        return argv[++i];
    };

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--help" || a == "-h") {
                print_help();
                return 0;
            } else if (a == "--config") {
                config_path = need(i, "--config");
            }
        }
        load_config(config_path, opt);
        if (fs::exists(config_path)) {
            resolve_option_paths(fs::path(config_path), opt);
            std::cout << "[config] loaded " << fs::path(config_path).lexically_normal().string() << "
";
        } else if (config_path != "vision_app.conf") {
            std::cerr << "[config] file not found: " << config_path << "
";
        }

        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--help" || a == "-h") {}
            else if (a == "--config") { ++i; }
            else if (a == "--mode") opt.mode = need(i, "--mode");
            else if (a == "--probe-task") opt.probe_task = need(i, "--probe-task");
            else if (a == "--roi-mode") opt.roi_mode = normalize_roi_mode(need(i, "--roi-mode"));
            else if (a == "--device") opt.device = need(i, "--device");
            else if (a == "--width") opt.width = std::stoi(need(i, "--width"));
            else if (a == "--height") opt.height = std::stoi(need(i, "--height"));
            else if (a == "--fps") opt.fps = std::stoi(need(i, "--fps"));
            else if (a == "--fourcc") opt.fourcc = need(i, "--fourcc");
            else if (a == "--buffer-size") opt.buffer_size = std::stoi(need(i, "--buffer-size"));
            else if (a == "--latest-only") opt.latest_only = parse_bool(need(i, "--latest-only"));
            else if (a == "--drain-grabs") opt.drain_grabs = std::stoi(need(i, "--drain-grabs"));
            else if (a == "--ui") opt.ui = parse_bool(need(i, "--ui"));
            else if (a == "--headless") opt.ui = false;
            else if (a == "--draw-overlay") opt.draw_overlay = parse_bool(need(i, "--draw-overlay"));
            else if (a == "--text-console") opt.text_console = parse_bool(need(i, "--text-console"));
            else if (a == "--duration") opt.duration = std::stoi(need(i, "--duration"));

            else if (a == "--camera-soft-max") opt.camera_soft_max = std::stoi(need(i, "--camera-soft-max"));
            else if (a == "--camera-preview-max") opt.camera_preview_max = std::stoi(need(i, "--camera-preview-max"));
            else if (a == "--warp-preview-max") opt.warp_preview_max = std::stoi(need(i, "--warp-preview-max"));

            else if (a == "--warp-width") opt.warp_width = std::stoi(need(i, "--warp-width"));
            else if (a == "--warp-height") opt.warp_height = std::stoi(need(i, "--warp-height"));
            else if (a == "--target-tag-px") opt.target_tag_px = std::stoi(need(i, "--target-tag-px"));
            else if (a == "--warp-center-x-ratio") opt.warp_center_x_ratio = std::stod(need(i, "--warp-center-x-ratio"));
            else if (a == "--warp-center-y-ratio") opt.warp_center_y_ratio = std::stod(need(i, "--warp-center-y-ratio"));
            else if (a == "--tag-family") opt.tag_family = need(i, "--tag-family");
            else if (a == "--target-id") opt.target_id = std::stoi(need(i, "--target-id"));
            else if (a == "--require-target-id") opt.require_target_id = parse_bool(need(i, "--require-target-id"));
            else if (a == "--manual-lock-only") opt.manual_lock_only = parse_bool(need(i, "--manual-lock-only"));
            else if (a == "--lock-frames") opt.lock_frames = std::stoi(need(i, "--lock-frames"));

            else if (a == "--save-warp") opt.save_warp = need(i, "--save-warp");
            else if (a == "--load-warp") opt.load_warp = need(i, "--load-warp");
            else if (a == "--save-rois") opt.save_rois = need(i, "--save-rois");
            else if (a == "--load-rois") opt.load_rois = need(i, "--load-rois");
            else if (a == "--save-report") opt.save_report = need(i, "--save-report");
            else if (a == "--snap-path") opt.snap_path = need(i, "--snap-path");

            else if (a == "--red-roi") {
                if (!parse_roi_csv(need(i, "--red-roi"), opt.default_rois.red_roi)) throw std::runtime_error("bad --red-roi");
            }
            else if (a == "--image-roi") {
                if (!parse_roi_csv(need(i, "--image-roi"), opt.default_rois.image_roi)) throw std::runtime_error("bad --image-roi");
            }

            else if (a == "--red-h1-low") opt.red_cfg.h1_low = std::stoi(need(i, "--red-h1-low"));
            else if (a == "--red-h1-high") opt.red_cfg.h1_high = std::stoi(need(i, "--red-h1-high"));
            else if (a == "--red-h2-low") opt.red_cfg.h2_low = std::stoi(need(i, "--red-h2-low"));
            else if (a == "--red-h2-high") opt.red_cfg.h2_high = std::stoi(need(i, "--red-h2-high"));
            else if (a == "--red-s-min") opt.red_cfg.s_min = std::stoi(need(i, "--red-s-min"));
            else if (a == "--red-v-min") opt.red_cfg.v_min = std::stoi(need(i, "--red-v-min"));

            else if (a == "--red-band-y0") opt.dynamic_red.band_y0 = std::stoi(need(i, "--red-band-y0"));
            else if (a == "--red-band-y1") opt.dynamic_red.band_y1 = std::stoi(need(i, "--red-band-y1"));
            else if (a == "--red-search-x0") opt.dynamic_red.search_x0 = std::stoi(need(i, "--red-search-x0"));
            else if (a == "--red-search-x1") opt.dynamic_red.search_x1 = std::stoi(need(i, "--red-search-x1"));
            else if (a == "--roi-gap-above-band") opt.dynamic_red.roi_gap_above_band = std::stoi(need(i, "--roi-gap-above-band"));
            else if (a == "--roi-anchor-y") opt.dynamic_red.roi_anchor_y = std::stoi(need(i, "--roi-anchor-y"));
            else if (a == "--roi-width") opt.dynamic_red.roi_width = std::stoi(need(i, "--roi-width"));
            else if (a == "--roi-height") opt.dynamic_red.roi_height = std::stoi(need(i, "--roi-height"));
            else if (a == "--red-min-area") opt.dynamic_red.min_area = std::stoi(need(i, "--red-min-area"));
            else if (a == "--red-max-area") opt.dynamic_red.max_area = std::stoi(need(i, "--red-max-area"));
            else if (a == "--red-morph-k") opt.dynamic_red.morph_k = std::stoi(need(i, "--red-morph-k"));
            else if (a == "--red-miss-tolerance") opt.dynamic_red.miss_tolerance = std::stoi(need(i, "--red-miss-tolerance"));
            else if (a == "--red-fallback-center-x") opt.dynamic_red.fallback_center_x = std::stoi(need(i, "--red-fallback-center-x"));
            else if (a == "--red-center-alpha") opt.dynamic_red.center_alpha = std::stod(need(i, "--red-center-alpha"));
            else if (a == "--red-show-mask-window") opt.dynamic_red.show_mask_window = parse_bool(need(i, "--red-show-mask-window"));
            else if (a == "--red-require-dual-zone") opt.dynamic_red.require_dual_zone = parse_bool(need(i, "--red-require-dual-zone"));
            else if (a == "--red-zone-gap-px") opt.dynamic_red.zone_gap_px = std::stoi(need(i, "--red-zone-gap-px"));
            else if (a == "--red-zone-min-pixels") opt.dynamic_red.zone_min_pixels = std::stoi(need(i, "--red-zone-min-pixels"));
            else if (a == "--red-zone-min-ratio") opt.dynamic_red.zone_min_ratio = std::stod(need(i, "--red-zone-min-ratio"));
            else if (a == "--red-band-min-pixels") opt.dynamic_red.band_min_pixels = std::stoi(need(i, "--red-band-min-pixels"));
            else if (a == "--red-band-min-ratio") opt.dynamic_red.band_min_ratio = std::stod(need(i, "--red-band-min-ratio"));

            else if (a == "--model-enable") opt.model_cfg.enable = parse_bool(need(i, "--model-enable"));
            else if (a == "--model-backend") opt.model_cfg.backend = need(i, "--model-backend");
            else if (a == "--model-onnx-path") opt.model_cfg.onnx_path = need(i, "--model-onnx-path");
            else if (a == "--model-ncnn-param-path") opt.model_cfg.ncnn_param_path = need(i, "--model-ncnn-param-path");
            else if (a == "--model-ncnn-bin-path") opt.model_cfg.ncnn_bin_path = need(i, "--model-ncnn-bin-path");
            else if (a == "--model-labels-path") opt.model_cfg.labels_path = need(i, "--model-labels-path");
            else if (a == "--model-input-width") opt.model_cfg.input_width = std::stoi(need(i, "--model-input-width"));
            else if (a == "--model-input-height") opt.model_cfg.input_height = std::stoi(need(i, "--model-input-height"));
            else if (a == "--model-preprocess") opt.model_cfg.preprocess = need(i, "--model-preprocess");
            else if (a == "--model-threads") opt.model_cfg.threads = std::stoi(need(i, "--model-threads"));
            else if (a == "--model-stride") opt.model_cfg.stride = std::stoi(need(i, "--model-stride"));
            else if (a == "--model-topk") opt.model_cfg.topk = std::stoi(need(i, "--model-topk"));
            else if (a == "--model-max-hz") opt.model_max_hz = std::stod(need(i, "--model-max-hz"));

            else if (a == "--run-red") opt.run_red = parse_bool(need(i, "--run-red"));
            else if (a == "--run-image-roi") opt.run_image_roi = parse_bool(need(i, "--run-image-roi"));
            else if (a == "--run-model") opt.run_model = parse_bool(need(i, "--run-model"));
            else if (a == "--save-image-roi-dir") opt.save_image_roi_dir = need(i, "--save-image-roi-dir");
            else if (a == "--save-red-roi-dir") opt.save_red_roi_dir = need(i, "--save-red-roi-dir");
            else if (a == "--save-every-n") opt.save_every_n = std::stoi(need(i, "--save-every-n"));

            else throw std::runtime_error("unknown argument: " + a);
        }

        std::string err;
        if (opt.mode == "probe") {
            if (!run_probe_mode(opt, err)) {
                std::cerr << "Probe failed: " << err << "\n";
                return 1;
            }
            return 0;
        }
        if (opt.mode == "calibrate" || opt.mode == "live") {
            if (!run_calibrate(opt, err)) {
                std::cerr << "Calibrate failed: " << err << "\n";
                return 1;
            }
            return 0;
        }
        if (opt.mode == "deploy") {
            if (!run_deploy(opt, err)) {
                std::cerr << "Deploy failed: " << err << "\n";
                return 1;
            }
            return 0;
        }

        std::cerr << "Unknown mode: " << opt.mode << "\n";
        print_help();
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
