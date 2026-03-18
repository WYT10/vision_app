
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "deploy.hpp"

using namespace vision_app;

static std::string trim(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
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
        const std::string v = trim(line.substr(pos + 1));
        if (k == "mode") o.mode = v;
        else if (k == "device") o.device = v;
        else if (k == "width") o.width = std::stoi(v);
        else if (k == "height") o.height = std::stoi(v);
        else if (k == "fps") o.fps = std::stoi(v);
        else if (k == "fourcc") o.fourcc = v;
        else if (k == "buffer_size") o.buffer_size = std::stoi(v);
        else if (k == "latest_only") o.latest_only = parse_bool(v);
        else if (k == "drain_grabs") o.drain_grabs = std::stoi(v);
        else if (k == "headless") o.headless = parse_bool(v);
        else if (k == "duration") o.duration = std::stoi(v);
        else if (k == "ui") o.ui = parse_bool(v);
        else if (k == "debug") o.debug = parse_bool(v);
        else if (k == "save_roi_frames") o.save_roi_frames = parse_bool(v);
        else if (k == "save_roi_dir") o.save_roi_dir = v;
        else if (k == "camera_soft_max") o.camera_soft_max = std::stoi(v);
        else if (k == "camera_preview_max") o.camera_preview_max = std::stoi(v);
        else if (k == "warp_preview_max") o.warp_preview_max = std::stoi(v);
        else if (k == "tag_family") o.tag_cfg.family = v;
        else if (k == "target_id") o.tag_cfg.target_id = std::stoi(v);
        else if (k == "require_target_id") o.tag_cfg.require_target_id = parse_bool(v);
        else if (k == "manual_lock_only") o.tag_cfg.manual_lock_only = parse_bool(v);
        else if (k == "lock_frames") o.tag_cfg.lock_frames = std::stoi(v);
        else if (k == "warp_width") o.warp_width = std::stoi(v);
        else if (k == "warp_height") o.warp_height = std::stoi(v);
        else if (k == "target_tag_px") o.target_tag_px = std::stoi(v);
        else if (k == "save_warp") o.save_warp = v;
        else if (k == "load_warp") o.load_warp = v;
        else if (k == "save_rois") o.save_rois = v;
        else if (k == "load_rois") o.load_rois = v;
        else if (k == "save_report") o.save_report = v;
        else if (k == "red_roi") parse_roi_csv(v, o.default_rois.red_roi);
        else if (k == "image_roi") parse_roi_csv(v, o.default_rois.image_roi);
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
    }
}

static void print_help() {
    std::cout
        << "vision_app modes: probe | calibrate | deploy\n\n"
        << "Core camera args\n"
        << "  --device /dev/video0\n"
        << "  --width 640 --height 480\n"
        << "  --fourcc MJPG\n"
        << "  --fps 120\n"
        << "  --buffer-size 1\n"
        << "  --latest-only 1\n"
        << "  --drain-grabs 1\n"
        << "  --headless 0\n"
        << "  --duration 5\n\n"
        << "Calibration args\n"
        << "  --tag-family auto|16|25|36\n"
        << "  --target-id 0\n"
        << "  --require-target-id 1\n"
        << "  --warp-width 384 --warp-height 384\n"
        << "  --target-tag-px 128\n"
        << "  --save-warp PATH --load-warp PATH\n"
        << "  --save-rois PATH --load-rois PATH\n"
        << "  --red-roi x,y,w,h --image-roi x,y,w,h\n\n"
        << "Model args\n"
        << "  --model-enable 1\n"
        << "  --model-backend onnx|ncnn|off\n"
        << "  --model-onnx-path PATH\n"
        << "  --model-ncnn-param-path PATH\n"
        << "  --model-ncnn-bin-path PATH\n"
        << "  --model-labels-path PATH\n"
        << "  --model-input-width 128 --model-input-height 128\n"
        << "  --model-preprocess crop|stretch\n"
        << "  --model-threads 4 --model-stride 5 --model-topk 5\n\n"
        << "UI / debug args\n"
        << "  --ui 1 --debug 1\n"
        << "  --camera-preview-max 640 --warp-preview-max 640\n"
        << "  --save-roi-frames 0 --save-roi-dir ../report/roi_snaps\n"
        << "  --save-report ../report/latest_report.md\n\n"
        << "Examples\n"
        << "  ./vision_app --mode probe --device /dev/video0 --width 640 --height 480 --fps 120\n"
        << "  ./vision_app --mode calibrate --device /dev/video0 --width 640 --height 480 --fps 120 --tag-family auto --target-id 0 --warp-width 384 --warp-height 384 --target-tag-px 128\n"
        << "  ./vision_app --mode deploy --device /dev/video0 --width 640 --height 480 --fps 120 --load-warp ../report/warp_package.yml.gz --load-rois ../report/rois.yml --model-enable 1 --model-backend ncnn --model-ncnn-param-path ../models/model.ncnn.param --model-ncnn-bin-path ../models/model.ncnn.bin --model-labels-path ../models/labels.txt\n";
}

int main(int argc, char** argv) {
    AppOptions opt;
    load_config("./vision_app.conf", opt);

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << name << "\n"; std::exit(1); }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { print_help(); return 0; }
        else if (a == "--mode") opt.mode = need("--mode");
        else if (a == "--device") opt.device = need("--device");
        else if (a == "--width") opt.width = std::stoi(need("--width"));
        else if (a == "--height") opt.height = std::stoi(need("--height"));
        else if (a == "--fps") opt.fps = std::stoi(need("--fps"));
        else if (a == "--fourcc") opt.fourcc = need("--fourcc");
        else if (a == "--buffer-size") opt.buffer_size = std::stoi(need("--buffer-size"));
        else if (a == "--latest-only") opt.latest_only = parse_bool(need("--latest-only"));
        else if (a == "--drain-grabs") opt.drain_grabs = std::stoi(need("--drain-grabs"));
        else if (a == "--headless") opt.headless = parse_bool(need("--headless"));
        else if (a == "--duration") opt.duration = std::stoi(need("--duration"));
        else if (a == "--ui") opt.ui = parse_bool(need("--ui"));
        else if (a == "--debug") opt.debug = parse_bool(need("--debug"));
        else if (a == "--save-roi-frames") opt.save_roi_frames = parse_bool(need("--save-roi-frames"));
        else if (a == "--save-roi-dir") opt.save_roi_dir = need("--save-roi-dir");
        else if (a == "--camera-soft-max") opt.camera_soft_max = std::stoi(need("--camera-soft-max"));
        else if (a == "--camera-preview-max") opt.camera_preview_max = std::stoi(need("--camera-preview-max"));
        else if (a == "--warp-preview-max") opt.warp_preview_max = std::stoi(need("--warp-preview-max"));
        else if (a == "--tag-family") opt.tag_cfg.family = need("--tag-family");
        else if (a == "--target-id") opt.tag_cfg.target_id = std::stoi(need("--target-id"));
        else if (a == "--require-target-id") opt.tag_cfg.require_target_id = parse_bool(need("--require-target-id"));
        else if (a == "--manual-lock-only") opt.tag_cfg.manual_lock_only = parse_bool(need("--manual-lock-only"));
        else if (a == "--lock-frames") opt.tag_cfg.lock_frames = std::stoi(need("--lock-frames"));
        else if (a == "--warp-width") opt.warp_width = std::stoi(need("--warp-width"));
        else if (a == "--warp-height") opt.warp_height = std::stoi(need("--warp-height"));
        else if (a == "--target-tag-px") opt.target_tag_px = std::stoi(need("--target-tag-px"));
        else if (a == "--save-warp") opt.save_warp = need("--save-warp");
        else if (a == "--load-warp") opt.load_warp = need("--load-warp");
        else if (a == "--save-rois") opt.save_rois = need("--save-rois");
        else if (a == "--load-rois") opt.load_rois = need("--load-rois");
        else if (a == "--save-report") opt.save_report = need("--save-report");
        else if (a == "--red-roi") { if (!parse_roi_csv(need("--red-roi"), opt.default_rois.red_roi)) { std::cerr << "bad --red-roi\n"; return 1; } }
        else if (a == "--image-roi") { if (!parse_roi_csv(need("--image-roi"), opt.default_rois.image_roi)) { std::cerr << "bad --image-roi\n"; return 1; } }
        else if (a == "--red-h1-low") opt.red_cfg.h1_low = std::stoi(need("--red-h1-low"));
        else if (a == "--red-h1-high") opt.red_cfg.h1_high = std::stoi(need("--red-h1-high"));
        else if (a == "--red-h2-low") opt.red_cfg.h2_low = std::stoi(need("--red-h2-low"));
        else if (a == "--red-h2-high") opt.red_cfg.h2_high = std::stoi(need("--red-h2-high"));
        else if (a == "--red-s-min") opt.red_cfg.s_min = std::stoi(need("--red-s-min"));
        else if (a == "--red-v-min") opt.red_cfg.v_min = std::stoi(need("--red-v-min"));
        else if (a == "--model-enable") opt.model_cfg.enable = parse_bool(need("--model-enable"));
        else if (a == "--model-backend") opt.model_cfg.backend = need("--model-backend");
        else if (a == "--model-onnx-path") opt.model_cfg.onnx_path = need("--model-onnx-path");
        else if (a == "--model-ncnn-param-path") opt.model_cfg.ncnn_param_path = need("--model-ncnn-param-path");
        else if (a == "--model-ncnn-bin-path") opt.model_cfg.ncnn_bin_path = need("--model-ncnn-bin-path");
        else if (a == "--model-labels-path") opt.model_cfg.labels_path = need("--model-labels-path");
        else if (a == "--model-input-width") opt.model_cfg.input_width = std::stoi(need("--model-input-width"));
        else if (a == "--model-input-height") opt.model_cfg.input_height = std::stoi(need("--model-input-height"));
        else if (a == "--model-preprocess") opt.model_cfg.preprocess = need("--model-preprocess");
        else if (a == "--model-threads") opt.model_cfg.threads = std::stoi(need("--model-threads"));
        else if (a == "--model-stride") opt.model_cfg.stride = std::stoi(need("--model-stride"));
        else if (a == "--model-topk") opt.model_cfg.topk = std::stoi(need("--model-topk"));
        else { std::cerr << "unknown argument: " << a << "\n"; return 1; }
    }

    std::string err;
    if (opt.mode == "probe") {
        if (!run_probe(opt, err)) { std::cerr << "Probe failed: " << err << "\n"; return 1; }
        return 0;
    }
    if (opt.mode == "calibrate") {
        if (!run_calibrate(opt, err)) { std::cerr << "Calibrate failed: " << err << "\n"; return 1; }
        return 0;
    }
    if (opt.mode == "deploy") {
        if (!run_deploy(opt, err)) { std::cerr << "Deploy failed: " << err << "\n"; return 1; }
        return 0;
    }

    std::cerr << "unknown mode: " << opt.mode << "\n";
    return 1;
}
