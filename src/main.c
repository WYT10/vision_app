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
        else if (k == "camera_soft_max") o.camera_soft_max = std::stoi(v);
        else if (k == "warp_soft_max") o.warp_soft_max = std::stoi(v);
        else if (k == "preview_soft_max") { o.preview_soft_max = std::stoi(v); o.camera_preview_max = o.preview_soft_max; o.warp_preview_max = o.preview_soft_max; }
        else if (k == "camera_preview_max") o.camera_preview_max = std::stoi(v);
        else if (k == "warp_preview_max") o.warp_preview_max = std::stoi(v);
        else if (k == "temp_preview_square") o.temp_preview_square = std::stoi(v);
        else if (k == "temp_preview_stride") o.temp_preview_stride = std::stoi(v);
        else if (k == "tag_fill_ratio") o.tag_fill_ratio = std::stod(v);
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
        else if (k == "model_path") o.model_cfg.path = v;
        else if (k == "model_input_width") o.model_cfg.input_width = std::stoi(v);
        else if (k == "model_input_height") o.model_cfg.input_height = std::stoi(v);
        else if (k == "model_stride") o.model_cfg.stride = std::stoi(v);
    }
}

static void print_help() {
    std::cout
        << "vision_app modes: probe | bench | live | deploy\n\n"
        << "Build every time with:\n"
        << "  mkdir -p build\n"
        << "  cd build\n"
        << "  cmake ..\n"
        << "  make -j$(nproc)\n\n"
        << "Core camera args\n"
        << "  --device /dev/video0\n"
        << "  --width 640 --height 480\n"
        << "  --fourcc MJPG\n"
        << "  --fps 180\n"
        << "  --buffer-size 1\n"
        << "  --latest-only 1\n"
        << "  --drain-grabs 1\n"
        << "  --headless 1\n"
        << "  --duration 10\n\n"
        << "Preview / safety\n"
        << "  --camera-soft-max 1000\n"
        << "  --warp-soft-max 700\n"
        << "  --camera-preview-max 640\n"
        << "  --warp-preview-max 640\n"
        << "  --temp-preview-square 260\n"
        << "  --temp-preview-stride 3\n\n"
        << "Tag args\n"
        << "  --tag-family auto|16|25|36\n"
        << "  --tag-fill-ratio 0.70\n"
        << "  --target-id 0\n"
        << "  --require-target-id 1\n"
        << "  --manual-lock-only 1\n"
        << "  --lock-frames 4\n\n"
        << "ROI args\n"
        << "  --red-roi x,y,w,h\n"
        << "  --image-roi x,y,w,h\n"
        << "  ratios are in warp space, 0..1\n\n"
        << "Red threshold args\n"
        << "  --red-h1-low 0 --red-h1-high 10\n"
        << "  --red-h2-low 170 --red-h2-high 180\n"
        << "  --red-s-min 80 --red-v-min 60\n\n"
        << "Model args\n"
        << "  --model-enable 1\n"
        << "  --model-backend onnx|ncnn\n"
        << "  --model-path ../models/model.onnx\n"
        << "  --model-input-width 224 --model-input-height 224\n"
        << "  --model-stride 5\n\n"
        << "Save / load\n"
        << "  --save-warp PATH   --load-warp PATH\n"
        << "  --save-rois PATH   --load-rois PATH\n"
        << "  --save-report PATH\n\n"
        << "Examples\n"
        << "  ./vision_app --mode probe\n"
        << "  ./vision_app --mode bench --device /dev/video0 --width 640 --height 480 --fourcc MJPG --fps 180 --buffer-size 1 --latest-only 1 --drain-grabs 1 --headless 1 --duration 10\n"
        << "  ./vision_app --mode live --device /dev/video0 --width 640 --height 480 --fourcc MJPG --fps 120 --buffer-size 1 --latest-only 1 --drain-grabs 1 --tag-family auto --target-id 0 --manual-lock-only 1 --tag-fill-ratio 0.70 --camera-preview-max 700 --warp-preview-max 700\n"
        << "  ./vision_app --mode deploy --device /dev/video0 --width 640 --height 480 --fourcc MJPG --fps 180 --load-warp ../report/warp_package.yml.gz --load-rois ../report/rois.yml --model-enable 1 --model-backend onnx --model-path ../models/model.onnx\n";
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
        else if (a == "--camera-soft-max") opt.camera_soft_max = std::stoi(need("--camera-soft-max"));
        else if (a == "--warp-soft-max") opt.warp_soft_max = std::stoi(need("--warp-soft-max"));
        else if (a == "--preview-soft-max") { opt.preview_soft_max = std::stoi(need("--preview-soft-max")); opt.camera_preview_max = opt.preview_soft_max; opt.warp_preview_max = opt.preview_soft_max; }
        else if (a == "--camera-preview-max") opt.camera_preview_max = std::stoi(need("--camera-preview-max"));
        else if (a == "--warp-preview-max") opt.warp_preview_max = std::stoi(need("--warp-preview-max"));
        else if (a == "--temp-preview-square") opt.temp_preview_square = std::stoi(need("--temp-preview-square"));
        else if (a == "--temp-preview-stride") opt.temp_preview_stride = std::stoi(need("--temp-preview-stride"));
        else if (a == "--tag-fill-ratio") opt.tag_fill_ratio = std::stod(need("--tag-fill-ratio"));
        else if (a == "--tag-family") opt.tag_family = need("--tag-family");
        else if (a == "--target-id") opt.target_id = std::stoi(need("--target-id"));
        else if (a == "--require-target-id") opt.require_target_id = parse_bool(need("--require-target-id"));
        else if (a == "--manual-lock-only") opt.manual_lock_only = parse_bool(need("--manual-lock-only"));
        else if (a == "--lock-frames") opt.lock_frames = std::stoi(need("--lock-frames"));
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
        else if (a == "--model-path") opt.model_cfg.path = need("--model-path");
        else if (a == "--model-input-width") opt.model_cfg.input_width = std::stoi(need("--model-input-width"));
        else if (a == "--model-input-height") opt.model_cfg.input_height = std::stoi(need("--model-input-height"));
        else if (a == "--model-stride") opt.model_cfg.stride = std::stoi(need("--model-stride"));
        else { std::cerr << "unknown argument: " << a << "\n"; return 1; }
    }

    std::string err;
    if (opt.mode == "probe") {
        CameraProbeResult probe;
        if (!probe_camera(opt.device, probe, err)) { std::cerr << "Probe failed: " << err << "\n"; return 1; }
        print_probe(probe);
        return 0;
    }
    if (opt.mode == "bench") {
        RuntimeStats stats;
        if (!bench_capture(opt.device, opt.width, opt.height, opt.fps, opt.fourcc, opt.buffer_size, opt.latest_only, opt.drain_grabs, opt.headless, opt.duration, opt.camera_soft_max, opt.camera_preview_max, stats, err)) {
            std::cerr << "Bench failed: " << err << "\n"; return 1;
        }
        print_runtime_stats(stats);
        return 0;
    }
    if (opt.mode == "live") {
        if (!run_live(opt, err)) { std::cerr << "Live failed: " << err << "\n"; return 1; }
        return 0;
    }
    if (opt.mode == "deploy") {
        if (!run_deploy(opt, err)) { std::cerr << "Deploy failed: " << err << "\n"; return 1; }
        return 0;
    }

    std::cerr << "unknown mode: " << opt.mode << "\n";
    return 1;
}
