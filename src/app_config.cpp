#include "app_config.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace vision_app {
namespace {

std::string trim(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

bool parse_bool(const std::string& v) {
    return v == "1" || v == "true" || v == "TRUE" || v == "yes" || v == "on";
}

std::string bool_str(bool v) { return v ? "1" : "0"; }

bool parse_roi_csv(const std::string& s, RoiRatio& r) {
    std::stringstream ss(s);
    std::string tok;
    double vals[4];
    int i = 0;
    while (std::getline(ss, tok, ',') && i < 4) vals[i++] = std::stod(trim(tok));
    if (i != 4) return false;
    r.x = vals[0]; r.y = vals[1]; r.w = vals[2]; r.h = vals[3];
    r = clamp_roi(r);
    return true;
}

bool parse_band_csv(const std::string& s, BandRatio& b) {
    std::stringstream ss(s);
    std::string tok;
    double vals[2];
    int i = 0;
    while (std::getline(ss, tok, ',') && i < 2) vals[i++] = std::stod(trim(tok));
    if (i != 2) return false;
    b.y = std::clamp(vals[0], 0.0, 0.98);
    b.h = std::clamp(vals[1], 0.01, 1.0 - b.y);
    return true;
}

std::string roi_csv(const RoiRatio& r) {
    std::ostringstream oss;
    oss << r.x << ',' << r.y << ',' << r.w << ',' << r.h;
    return oss.str();
}

std::string band_csv(const BandRatio& b) {
    std::ostringstream oss;
    oss << b.y << ',' << b.h;
    return oss.str();
}

bool is_probably_stream_path(const std::string& s) {
    return s.rfind("rtsp://", 0) == 0 || s.rfind("http://", 0) == 0 || s.rfind("https://", 0) == 0;
}

std::filesystem::path materialize_config_path(const std::string& raw) {
    if (raw.empty()) return {};
    std::filesystem::path p(raw);
    if (p.is_absolute()) return p;
    if (std::filesystem::exists(p)) return std::filesystem::absolute(p);

    const auto cwd = std::filesystem::current_path();
    if (cwd.filename() == "build") {
        const auto candidate = cwd.parent_path() / p;
        if (std::filesystem::exists(candidate)) return candidate;
    }
    return std::filesystem::absolute(p);
}

std::filesystem::path choose_project_root_from_config(const std::filesystem::path& config_path) {
    if (config_path.empty()) return std::filesystem::current_path();
    std::filesystem::path abs_config = config_path;
    if (!abs_config.is_absolute()) abs_config = std::filesystem::absolute(abs_config);
    const std::filesystem::path config_dir = abs_config.parent_path();
    if (config_dir.filename() == "config" && config_dir.has_parent_path()) return config_dir.parent_path();
    return config_dir.empty() ? std::filesystem::current_path() : config_dir;
}

std::string resolve_file_like_path(const std::string& raw, const std::filesystem::path& project_root) {
    if (raw.empty() || is_probably_stream_path(raw)) return raw;
    std::filesystem::path p(raw);
    if (p.is_absolute()) return p.lexically_normal().string();
    return (project_root / p).lexically_normal().string();
}

void ensure_parent_dir(const std::filesystem::path& p) {
    const auto parent = p.parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
}

} // namespace

bool load_profile_config(const std::string& path, AppOptions& o, std::string& err) {
    err.clear();
    const std::filesystem::path config_fs_path = materialize_config_path(path);
    std::ifstream in(config_fs_path);
    if (!in.is_open()) return true;

    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        const auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        const std::string k = trim(line.substr(0, pos));
        const std::string v = trim(line.substr(pos + 1));

        if (k == "mode") o.mode = v;
        else if (k == "profile_path") o.profile_path = v;
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
        else if (k == "camera_soft_max") o.camera_soft_max = std::stoi(v);
        else if (k == "camera_preview_max") o.camera_preview_max = std::stoi(v);
        else if (k == "warp_preview_max") o.warp_preview_max = std::stoi(v);
        else if (k == "status_width") o.status_width = std::stoi(v);
        else if (k == "show_status_window") o.show_status_window = parse_bool(v);
        else if (k == "text_sink") o.text_sink = v;
        else if (k == "warp_width") o.warp_width = std::stoi(v);
        else if (k == "warp_height") o.warp_height = std::stoi(v);
        else if (k == "target_tag_px") o.target_tag_px = std::stoi(v);
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
        else if (k == "trigger_mode") o.trigger_mode = v;
        else if (k == "red_roi") parse_roi_csv(v, o.default_rois.red_roi);
        else if (k == "image_roi") parse_roi_csv(v, o.default_rois.image_roi);
        else if (k == "red_ratio_threshold") o.fixed_cfg.red_ratio_threshold = std::stod(v);
        else if (k == "upper_band") parse_band_csv(v, o.dynamic_cfg.upper_band);
        else if (k == "lower_band") parse_band_csv(v, o.dynamic_cfg.lower_band);
        else if (k == "min_red_width_ratio") o.dynamic_cfg.min_red_width_ratio = std::stod(v);
        else if (k == "min_red_fill_ratio") o.dynamic_cfg.min_red_fill_ratio = std::stod(v);
        else if (k == "x_smoothing_alpha") o.dynamic_cfg.x_smoothing_alpha = std::stod(v);
        else if (k == "image_bottom_offset") o.dynamic_cfg.image_roi.bottom_offset = std::stod(v);
        else if (k == "dynamic_image_width") o.dynamic_cfg.image_roi.width = std::stod(v);
        else if (k == "dynamic_image_height") o.dynamic_cfg.image_roi.height = std::stod(v);
        else if (k == "red_h1_low") o.red_cfg.h1_low = std::stoi(v);
        else if (k == "red_h1_high") o.red_cfg.h1_high = std::stoi(v);
        else if (k == "red_h2_low") o.red_cfg.h2_low = std::stoi(v);
        else if (k == "red_h2_high") o.red_cfg.h2_high = std::stoi(v);
        else if (k == "red_s_min") o.red_cfg.s_min = std::stoi(v);
        else if (k == "red_v_min") o.red_cfg.v_min = std::stoi(v);
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
        else if (k == "automation_enable") o.automation_enable = parse_bool(v);
        else if (k == "automation_mode") o.automation_mode = v;
        else if (k == "automation_server_url") o.automation_server_url = v;
        else if (k == "automation_session") o.automation_session = v;
        else if (k == "automation_collect_dir") o.automation_collect_dir = v;
        else if (k == "automation_poll_ms") o.automation_poll_ms = std::stoi(v);
        else if (k == "automation_settle_ms") o.automation_settle_ms = std::stoi(v);
        else if (k == "automation_max_per_trial") o.automation_max_per_trial = std::stoi(v);
        else if (k == "automation_wrong_only") o.automation_wrong_only = parse_bool(v);
        else if (k == "automation_low_conf_threshold") o.automation_low_conf_threshold = std::stod(v);
    }
    return true;
}

bool save_profile_config(const std::string& path, const AppOptions& o, std::string& err) {
    err.clear();
    ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out.is_open()) {
        err = "cannot write profile: " + path;
        return false;
    }

    out
        << "# vision_app profile\n"
        << "mode=" << o.mode << "\n"
        << "profile_path=" << o.profile_path << "\n"
        << "device=" << o.device << "\n"
        << "width=" << o.width << "\n"
        << "height=" << o.height << "\n"
        << "fps=" << o.fps << "\n"
        << "fourcc=" << o.fourcc << "\n"
        << "buffer_size=" << o.buffer_size << "\n"
        << "latest_only=" << bool_str(o.latest_only) << "\n"
        << "drain_grabs=" << o.drain_grabs << "\n"
        << "ui=" << bool_str(o.ui) << "\n"
        << "draw_overlay=" << bool_str(o.draw_overlay) << "\n"
        << "duration=" << o.duration << "\n"
        << "camera_soft_max=" << o.camera_soft_max << "\n"
        << "camera_preview_max=" << o.camera_preview_max << "\n"
        << "warp_preview_max=" << o.warp_preview_max << "\n"
        << "status_width=" << o.status_width << "\n"
        << "show_status_window=" << bool_str(o.show_status_window) << "\n"
        << "text_sink=" << o.text_sink << "\n"
        << "warp_width=" << o.warp_width << "\n"
        << "warp_height=" << o.warp_height << "\n"
        << "target_tag_px=" << o.target_tag_px << "\n"
        << "tag_family=" << o.tag_family << "\n"
        << "target_id=" << o.target_id << "\n"
        << "require_target_id=" << bool_str(o.require_target_id) << "\n"
        << "manual_lock_only=" << bool_str(o.manual_lock_only) << "\n"
        << "lock_frames=" << o.lock_frames << "\n"
        << "save_warp=" << o.save_warp << "\n"
        << "load_warp=" << o.load_warp << "\n"
        << "save_rois=" << o.save_rois << "\n"
        << "load_rois=" << o.load_rois << "\n"
        << "save_report=" << o.save_report << "\n"
        << "trigger_mode=" << o.trigger_mode << "\n"
        << "red_roi=" << roi_csv(o.default_rois.red_roi) << "\n"
        << "image_roi=" << roi_csv(o.default_rois.image_roi) << "\n"
        << "red_ratio_threshold=" << o.fixed_cfg.red_ratio_threshold << "\n"
        << "upper_band=" << band_csv(o.dynamic_cfg.upper_band) << "\n"
        << "lower_band=" << band_csv(o.dynamic_cfg.lower_band) << "\n"
        << "min_red_width_ratio=" << o.dynamic_cfg.min_red_width_ratio << "\n"
        << "min_red_fill_ratio=" << o.dynamic_cfg.min_red_fill_ratio << "\n"
        << "x_smoothing_alpha=" << o.dynamic_cfg.x_smoothing_alpha << "\n"
        << "image_bottom_offset=" << o.dynamic_cfg.image_roi.bottom_offset << "\n"
        << "dynamic_image_width=" << o.dynamic_cfg.image_roi.width << "\n"
        << "dynamic_image_height=" << o.dynamic_cfg.image_roi.height << "\n"
        << "red_h1_low=" << o.red_cfg.h1_low << "\n"
        << "red_h1_high=" << o.red_cfg.h1_high << "\n"
        << "red_h2_low=" << o.red_cfg.h2_low << "\n"
        << "red_h2_high=" << o.red_cfg.h2_high << "\n"
        << "red_s_min=" << o.red_cfg.s_min << "\n"
        << "red_v_min=" << o.red_cfg.v_min << "\n"
        << "model_enable=" << bool_str(o.model_cfg.enable) << "\n"
        << "model_backend=" << o.model_cfg.backend << "\n"
        << "model_onnx_path=" << o.model_cfg.onnx_path << "\n"
        << "model_ncnn_param_path=" << o.model_cfg.ncnn_param_path << "\n"
        << "model_ncnn_bin_path=" << o.model_cfg.ncnn_bin_path << "\n"
        << "model_labels_path=" << o.model_cfg.labels_path << "\n"
        << "model_input_width=" << o.model_cfg.input_width << "\n"
        << "model_input_height=" << o.model_cfg.input_height << "\n"
        << "model_preprocess=" << o.model_cfg.preprocess << "\n"
        << "model_threads=" << o.model_cfg.threads << "\n"
        << "model_stride=" << o.model_cfg.stride << "\n"
        << "model_topk=" << o.model_cfg.topk << "\n"
        << "model_max_hz=" << o.model_max_hz << "\n"
        << "run_red=" << bool_str(o.run_red) << "\n"
        << "run_image_roi=" << bool_str(o.run_image_roi) << "\n"
        << "run_model=" << bool_str(o.run_model) << "\n"
        << "save_image_roi_dir=" << o.save_image_roi_dir << "\n"
        << "save_red_roi_dir=" << o.save_red_roi_dir << "\n"
        << "save_every_n=" << o.save_every_n << "\n"
        << "automation_enable=" << bool_str(o.automation_enable) << "\n"
        << "automation_mode=" << o.automation_mode << "\n"
        << "automation_server_url=" << o.automation_server_url << "\n"
        << "automation_session=" << o.automation_session << "\n"
        << "automation_collect_dir=" << o.automation_collect_dir << "\n"
        << "automation_poll_ms=" << o.automation_poll_ms << "\n"
        << "automation_settle_ms=" << o.automation_settle_ms << "\n"
        << "automation_max_per_trial=" << o.automation_max_per_trial << "\n"
        << "automation_wrong_only=" << bool_str(o.automation_wrong_only) << "\n"
        << "automation_low_conf_threshold=" << o.automation_low_conf_threshold << "\n";
    return true;
}

void resolve_profile_paths(const std::string& config_path, AppOptions& opt) {
    const std::filesystem::path concrete_config = materialize_config_path(config_path.empty() ? opt.profile_path : config_path);
    const std::filesystem::path project_root = choose_project_root_from_config(concrete_config);

    opt.profile_path = concrete_config.lexically_normal().string();
    opt.save_warp = resolve_file_like_path(opt.save_warp, project_root);
    opt.load_warp = resolve_file_like_path(opt.load_warp, project_root);
    opt.save_rois = resolve_file_like_path(opt.save_rois, project_root);
    opt.load_rois = resolve_file_like_path(opt.load_rois, project_root);
    opt.save_report = resolve_file_like_path(opt.save_report, project_root);
    opt.save_image_roi_dir = resolve_file_like_path(opt.save_image_roi_dir, project_root);
    opt.save_red_roi_dir = resolve_file_like_path(opt.save_red_roi_dir, project_root);
    opt.automation_collect_dir = resolve_file_like_path(opt.automation_collect_dir, project_root);
    opt.model_cfg.onnx_path = resolve_file_like_path(opt.model_cfg.onnx_path, project_root);
    opt.model_cfg.ncnn_param_path = resolve_file_like_path(opt.model_cfg.ncnn_param_path, project_root);
    opt.model_cfg.ncnn_bin_path = resolve_file_like_path(opt.model_cfg.ncnn_bin_path, project_root);
    opt.model_cfg.labels_path = resolve_file_like_path(opt.model_cfg.labels_path, project_root);
}

void print_help() {
    std::cout << R"TXT(
vision_app

Build:
  export CMAKE_PREFIX_PATH=/home/pi/ncnn/build/install:$CMAKE_PREFIX_PATH

  cd ~/Desktop/vision_app
  rm -rf build
  mkdir -p build
  cd build
  cmake ..
  make -j$(nproc)

Modes:
  --mode probe
  --mode live
  --mode calibrate
  --mode deploy

Core args:
  --config PATH
  --device /dev/video0 | rtsp://... | http://...
  --width 160 --height 120 --fourcc MJPG --fps 120
  --buffer-size 1
  --latest-only 1
  --drain-grabs 1
  --ui 1
  --draw-overlay 1
  --duration 10
  --camera-soft-max 1000
  --camera-preview-max 800
  --warp-preview-max 900
  --status-width 720

Tag / warp args:
  --warp-width 384 --warp-height 384
  --target-tag-px 128
  --tag-family auto|16|25|36
  --target-id 0
  --require-target-id 1
  --manual-lock-only 1
  --lock-frames 4
  --save-warp PATH
  --load-warp PATH
  --save-rois PATH
  --load-rois PATH
  --save-report PATH

Trigger args:
  --trigger-mode fixed_rect|dynamic_red_stacked
  --red-roi x,y,w,h
  --image-roi x,y,w,h
  --upper-band y,h
  --lower-band y,h
  --image-bottom-offset V
  --dynamic-image-width V
  --dynamic-image-height V
  --red-ratio-threshold V
  --min-red-width-ratio V
  --min-red-fill-ratio V

Red threshold args:
  --red-h1-low N
  --red-h1-high N
  --red-h2-low N
  --red-h2-high N
  --red-s-min N
  --red-v-min N

Model args:
  --model-enable 1
  --model-backend onnx|ncnn
  --model-onnx-path PATH
  --model-ncnn-param-path PATH
  --model-ncnn-bin-path PATH
  --model-labels-path PATH
  --model-input-width 128
  --model-input-height 128
  --model-preprocess crop|stretch|letterbox
  --model-threads 4
  --model-stride 1
  --model-topk 5
  --model-max-hz 5.0

Runtime save / partial pipeline:
  --run-red 1
  --run-image-roi 1
  --run-model 1
  --save-image-roi-dir PATH
  --save-red-roi-dir PATH
  --save-every-n 10

Status / UI:
  --text-sink overlay|status_window|terminal|split
  --show-status-window 1

Automation:
  --automation-enable 1
  --automation-mode demo|collect_retrain
  --automation-server-url http://LAPTOP_IP:8787
  --automation-session demo
  --automation-collect-dir report/automation
  --automation-poll-ms 250
  --automation-settle-ms 700
  --automation-max-per-trial 1
  --automation-wrong-only 1
  --automation-low-conf-threshold 0.65

Notes:
  - For probe, /dev/video* uses v4l2-ctl enumeration.
  - RTSP / HTTP sources fall back to OpenCV probe and show observed stream properties.
  - Relative save/load/model paths from the profile are resolved from project root when the profile is in ./config.
  - In calibrate: o=save profile only, y=save warp only, p=save profile+rois+warp+report.

Examples:
  ./vision_app --mode probe --device /dev/video0
  ./vision_app --mode probe --device rtsp://192.168.0.233:5500/camera
  ./vision_app --mode live --device /dev/video0 --width 160 --height 120 --fps 120
  ./vision_app --mode calibrate --config /home/pi/Desktop/vision_app/config/profile.conf
  ./vision_app --mode deploy --config /home/pi/Desktop/vision_app/config/profile.conf
  ./vision_app --mode deploy --config /home/pi/Desktop/vision_app/config/profile.conf --automation-enable 1 --automation-server-url http://192.168.0.10:8787
)TXT";
}

} // namespace vision_app
