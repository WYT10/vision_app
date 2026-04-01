#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

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

static std::string strip_comment(const std::string& s) {
    bool in_quotes = false;
    std::string out;
    for (char ch : s) {
        if (ch == '"') in_quotes = !in_quotes;
        if (!in_quotes && ch == '#') break;
        out.push_back(ch);
    }
    return trim(out);
}

static bool parse_bool(const std::string& v) {
    return v == "1" || v == "true" || v == "TRUE" || v == "yes" || v == "on";
}

static bool parse_roi_csv(const std::string& s, RoiRatio& r) {
    std::stringstream ss(s); std::string tok; double vals[4]; int i = 0;
    while (std::getline(ss, tok, ',') && i < 4) vals[i++] = std::stod(trim(tok));
    if (i != 4) return false;
    r.x = vals[0]; r.y = vals[1]; r.w = vals[2]; r.h = vals[3];
    r = clamp_roi(r);
    return true;
}

static std::string resolve_path(const fs::path& cfg_dir, const std::string& v) {
    if (v.empty()) return v;
    fs::path p(v);
    if (p.is_absolute()) return p.string();
    return (cfg_dir / p).lexically_normal().string();
}

static void load_config(const std::string& path, AppOptions& o) {
    std::ifstream in(path);
    if (!in.is_open()) return;
    const fs::path cfg_dir = fs::path(path).parent_path().empty() ? fs::current_path() : fs::path(path).parent_path();
    std::string line;
    while (std::getline(in, line)) {
        line = strip_comment(line);
        if (line.empty()) continue;
        const auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        const std::string k = trim(line.substr(0, pos));
        const std::string v = trim(line.substr(pos + 1));

        if (k == "mode") o.mode = v;
        else if (k == "probe_task") o.probe_task = v;
        else if (k == "roi_mode") o.roi_mode = v;
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
        else if (k == "text_console") o.text_console = parse_bool(v);
        else if (k == "red_show_mask_window") o.red_show_mask_window = parse_bool(v);
        else if (k == "duration") o.duration = std::stoi(v);
        else if (k == "snap_path") o.snap_path = resolve_path(cfg_dir, v);

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

        else if (k == "save_warp") o.save_warp = resolve_path(cfg_dir, v);
        else if (k == "load_warp") o.load_warp = resolve_path(cfg_dir, v);
        else if (k == "save_rois") o.save_rois = resolve_path(cfg_dir, v);
        else if (k == "load_rois") o.load_rois = resolve_path(cfg_dir, v);
        else if (k == "save_report") o.save_report = resolve_path(cfg_dir, v);

        else if (k == "red_roi") parse_roi_csv(v, o.default_rois.red_roi);
        else if (k == "image_roi") parse_roi_csv(v, o.default_rois.image_roi);

        else if (k == "red_h1_low") o.red_cfg.h1_low = std::stoi(v);
        else if (k == "red_h1_high") o.red_cfg.h1_high = std::stoi(v);
        else if (k == "red_h2_low") o.red_cfg.h2_low = std::stoi(v);
        else if (k == "red_h2_high") o.red_cfg.h2_high = std::stoi(v);
        else if (k == "red_s_min") o.red_cfg.s_min = std::stoi(v);
        else if (k == "red_v_min") o.red_cfg.v_min = std::stoi(v);
        else if (k == "red_morph_k") o.red_cfg.morph_k = std::stoi(v);

        else if (k == "dyn_search_x0") o.dyn_cfg.search_x0 = std::stoi(v);
        else if (k == "dyn_search_x1") o.dyn_cfg.search_x1 = std::stoi(v);
        else if (k == "dyn_upper_y0") o.dyn_cfg.upper_y0 = std::stoi(v);
        else if (k == "dyn_upper_y1") o.dyn_cfg.upper_y1 = std::stoi(v);
        else if (k == "dyn_lower_y0") o.dyn_cfg.lower_y0 = std::stoi(v);
        else if (k == "dyn_lower_y1") o.dyn_cfg.lower_y1 = std::stoi(v);
        else if (k == "dyn_zone_min_pixels") o.dyn_cfg.zone_min_pixels = std::stoi(v);
        else if (k == "dyn_zone_min_ratio") o.dyn_cfg.zone_min_ratio = std::stod(v);
        else if (k == "dyn_center_x_max_diff") o.dyn_cfg.center_x_max_diff = std::stoi(v);
        else if (k == "dyn_stable_frames_required") o.dyn_cfg.stable_frames_required = std::stoi(v);
        else if (k == "dyn_roi_width") o.dyn_cfg.roi_width = std::stoi(v);
        else if (k == "dyn_roi_height") o.dyn_cfg.roi_height = std::stoi(v);
        else if (k == "dyn_roi_gap_above_upper_zone") o.dyn_cfg.roi_gap_above_upper_zone = std::stoi(v);

        else if (k == "model_enable") o.model_cfg.enable = parse_bool(v);
        else if (k == "model_backend") o.model_cfg.backend = v;
        else if (k == "model_onnx_path") o.model_cfg.onnx_path = resolve_path(cfg_dir, v);
        else if (k == "model_ncnn_param_path") o.model_cfg.ncnn_param_path = resolve_path(cfg_dir, v);
        else if (k == "model_ncnn_bin_path") o.model_cfg.ncnn_bin_path = resolve_path(cfg_dir, v);
        else if (k == "model_labels_path") o.model_cfg.labels_path = resolve_path(cfg_dir, v);
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
        else if (k == "save_image_roi_dir") o.save_image_roi_dir = resolve_path(cfg_dir, v);
        else if (k == "save_red_roi_dir") o.save_red_roi_dir = resolve_path(cfg_dir, v);
        else if (k == "save_every_n") o.save_every_n = std::stoi(v);
    }
}

static void print_help() {
    std::cout << R"TXT(
vision_app completed detailed v2

Build:
  export CMAKE_PREFIX_PATH=/home/pi/ncnn/build/install:$CMAKE_PREFIX_PATH
  mkdir -p build && cd build
  cmake ..
  make -j$(nproc)

Modes:
  --mode probe --probe-task list|live|snap|bench
  --mode calibrate
  --mode deploy

Core:
  --config PATH
  --device /dev/video0
  --width 160 --height 120 --fourcc MJPG --fps 120
  --ui 1 --draw-overlay 1 --text-console 1 --red-show-mask-window 1

ROI modes:
  --roi-mode fixed
  --roi-mode dynamic_red_stacked

Dynamic stacked trigger:
  --dyn-search-x0 N --dyn-search-x1 N
  --dyn-upper-y0 N --dyn-upper-y1 N
  --dyn-lower-y0 N --dyn-lower-y1 N
  --dyn-zone-min-pixels N
  --dyn-zone-min-ratio F
  --dyn-center-x-max-diff N
  --dyn-stable-frames-required N
  --dyn-roi-width N --dyn-roi-height N
  --dyn-roi-gap-above-upper-zone N

Warp positioning:
  --warp-center-x-ratio F
  --warp-center-y-ratio F

Examples:
  ./vision_app --mode probe --probe-task list --device /dev/video0
  ./vision_app --mode probe --probe-task bench --device /dev/video0 --duration 5
  ./vision_app --mode calibrate --roi-mode dynamic_red_stacked --red-show-mask-window 1
  ./vision_app --mode deploy --config ../vision_app.conf --roi-mode dynamic_red_stacked
)TXT";
}

int main(int argc, char** argv) {
    AppOptions opt;
    std::string config_path = "vision_app.conf";

    auto need = [&](int& i, const char* flag)->std::string {
        if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
        return argv[++i];
    };

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--help" || a == "-h") { print_help(); return 0; }
            else if (a == "--config") config_path = need(i, "--config");
        }
        load_config(config_path, opt);

        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--help" || a == "-h") {}
            else if (a == "--config") { ++i; }
            else if (a == "--mode") opt.mode = need(i, "--mode");
            else if (a == "--probe-task") opt.probe_task = need(i, "--probe-task");
            else if (a == "--roi-mode") opt.roi_mode = need(i, "--roi-mode");
            else if (a == "--device") opt.device = need(i, "--device");
            else if (a == "--width") opt.width = std::stoi(need(i, "--width"));
            else if (a == "--height") opt.height = std::stoi(need(i, "--height"));
            else if (a == "--fps") opt.fps = std::stoi(need(i, "--fps"));
            else if (a == "--fourcc") opt.fourcc = need(i, "--fourcc");
            else if (a == "--buffer-size") opt.buffer_size = std::stoi(need(i, "--buffer-size"));
            else if (a == "--latest-only") opt.latest_only = parse_bool(need(i, "--latest-only"));
            else if (a == "--drain-grabs") opt.drain_grabs = std::stoi(need(i, "--drain-grabs"));
            else if (a == "--ui") opt.ui = parse_bool(need(i, "--ui"));
            else if (a == "--draw-overlay") opt.draw_overlay = parse_bool(need(i, "--draw-overlay"));
            else if (a == "--text-console") opt.text_console = parse_bool(need(i, "--text-console"));
            else if (a == "--red-show-mask-window") opt.red_show_mask_window = parse_bool(need(i, "--red-show-mask-window"));
            else if (a == "--duration") opt.duration = std::stoi(need(i, "--duration"));
            else if (a == "--snap-path") opt.snap_path = need(i, "--snap-path");
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
            else if (a == "--red-roi") { if (!parse_roi_csv(need(i, "--red-roi"), opt.default_rois.red_roi)) throw std::runtime_error("bad --red-roi"); }
            else if (a == "--image-roi") { if (!parse_roi_csv(need(i, "--image-roi"), opt.default_rois.image_roi)) throw std::runtime_error("bad --image-roi"); }
            else if (a == "--red-h1-low") opt.red_cfg.h1_low = std::stoi(need(i, "--red-h1-low"));
            else if (a == "--red-h1-high") opt.red_cfg.h1_high = std::stoi(need(i, "--red-h1-high"));
            else if (a == "--red-h2-low") opt.red_cfg.h2_low = std::stoi(need(i, "--red-h2-low"));
            else if (a == "--red-h2-high") opt.red_cfg.h2_high = std::stoi(need(i, "--red-h2-high"));
            else if (a == "--red-s-min") opt.red_cfg.s_min = std::stoi(need(i, "--red-s-min"));
            else if (a == "--red-v-min") opt.red_cfg.v_min = std::stoi(need(i, "--red-v-min"));
            else if (a == "--red-morph-k") opt.red_cfg.morph_k = std::stoi(need(i, "--red-morph-k"));
            else if (a == "--dyn-search-x0") opt.dyn_cfg.search_x0 = std::stoi(need(i, "--dyn-search-x0"));
            else if (a == "--dyn-search-x1") opt.dyn_cfg.search_x1 = std::stoi(need(i, "--dyn-search-x1"));
            else if (a == "--dyn-upper-y0") opt.dyn_cfg.upper_y0 = std::stoi(need(i, "--dyn-upper-y0"));
            else if (a == "--dyn-upper-y1") opt.dyn_cfg.upper_y1 = std::stoi(need(i, "--dyn-upper-y1"));
            else if (a == "--dyn-lower-y0") opt.dyn_cfg.lower_y0 = std::stoi(need(i, "--dyn-lower-y0"));
            else if (a == "--dyn-lower-y1") opt.dyn_cfg.lower_y1 = std::stoi(need(i, "--dyn-lower-y1"));
            else if (a == "--dyn-zone-min-pixels") opt.dyn_cfg.zone_min_pixels = std::stoi(need(i, "--dyn-zone-min-pixels"));
            else if (a == "--dyn-zone-min-ratio") opt.dyn_cfg.zone_min_ratio = std::stod(need(i, "--dyn-zone-min-ratio"));
            else if (a == "--dyn-center-x-max-diff") opt.dyn_cfg.center_x_max_diff = std::stoi(need(i, "--dyn-center-x-max-diff"));
            else if (a == "--dyn-stable-frames-required") opt.dyn_cfg.stable_frames_required = std::stoi(need(i, "--dyn-stable-frames-required"));
            else if (a == "--dyn-roi-width") opt.dyn_cfg.roi_width = std::stoi(need(i, "--dyn-roi-width"));
            else if (a == "--dyn-roi-height") opt.dyn_cfg.roi_height = std::stoi(need(i, "--dyn-roi-height"));
            else if (a == "--dyn-roi-gap-above-upper-zone") opt.dyn_cfg.roi_gap_above_upper_zone = std::stoi(need(i, "--dyn-roi-gap-above-upper-zone"));
            else if (a == "--run-red") opt.run_red = parse_bool(need(i, "--run-red"));
            else if (a == "--run-image-roi") opt.run_image_roi = parse_bool(need(i, "--run-image-roi"));
            else if (a == "--run-model") opt.run_model = parse_bool(need(i, "--run-model"));
            else if (a == "--save-image-roi-dir") opt.save_image_roi_dir = need(i, "--save-image-roi-dir");
            else if (a == "--save-red-roi-dir") opt.save_red_roi_dir = need(i, "--save-red-roi-dir");
            else if (a == "--save-every-n") opt.save_every_n = std::stoi(need(i, "--save-every-n"));
            else throw std::runtime_error("unknown argument: " + a);
        }

        clamp_dynamic_cfg(opt.dyn_cfg, {opt.warp_width, opt.warp_height});
        std::cout << build_effective_config(opt) << std::endl;

        std::string err;
        if (opt.mode == "probe") return run_probe(opt, err) ? 0 : (std::cerr << "Probe failed: " << err << "\n", 1);
        if (opt.mode == "calibrate" || opt.mode == "live") return run_calibrate(opt, err) ? 0 : (std::cerr << "Calibrate failed: " << err << "\n", 1);
        if (opt.mode == "deploy") return run_deploy(opt, err) ? 0 : (std::cerr << "Deploy failed: " << err << "\n", 1);

        std::cerr << "Unknown mode: " << opt.mode << "\n";
        print_help();
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
