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
        else if (k == "preview_soft_max") o.preview_soft_max = std::stoi(v);
        else if (k == "temp_preview_square") o.temp_preview_square = std::stoi(v);
        else if (k == "temp_preview_stride") o.temp_preview_stride = std::stoi(v);
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
    }
}

static void print_help() {
    std::cout << "vision_app modes: probe | bench | live | deploy\n\n";
    std::cout << "Core camera args:\n";
    std::cout << "  --device /dev/video0\n";
    std::cout << "  --width 640 --height 480\n";
    std::cout << "  --fourcc MJPG\n";
    std::cout << "  --fps 180\n";
    std::cout << "  --buffer-size 1        small buffer, fresher frames\n";
    std::cout << "  --latest-only 1        prefer latest frame\n";
    std::cout << "  --drain-grabs 1        drop queued stale frames\n";
    std::cout << "  --headless 1           no UI, best for pure bench\n";
    std::cout << "  --duration 10          bench duration in seconds\n\n";
    std::cout << "Live/deploy safety args:\n";
    std::cout << "  --camera-soft-max 1000 clamp camera request if too large\n";
    std::cout << "  --warp-soft-max 700    clamp fitted warp size for Pi safety\n";
    std::cout << "  --preview-soft-max 500 downscale display preview only\n";
    std::cout << "  --temp-preview-square 220 small live warp inset before lock\n";
    std::cout << "  --temp-preview-stride 3 update inset every N frames\n\n";
    std::cout << "AprilTag args:\n";
    std::cout << "  --tag-family auto|16|25|36\n";
    std::cout << "  --target-id 0\n";
    std::cout << "  --require-target-id 1\n";
    std::cout << "  --manual-lock-only 1\n";
    std::cout << "  --lock-frames 4\n\n";
    std::cout << "ROI / save-load args:\n";
    std::cout << "  --red-roi x,y,w,h\n";
    std::cout << "  --image-roi x,y,w,h\n";
    std::cout << "  --save-warp PATH  --load-warp PATH\n";
    std::cout << "  --save-rois PATH  --load-rois PATH\n";
    std::cout << "  --save-report PATH\n\n";
    std::cout << "Useful examples:\n";
    std::cout << "  ./vision_app --mode probe\n";
    std::cout << "  ./vision_app --mode bench --device /dev/video0 --width 640 --height 480 --fourcc MJPG --fps 180 --buffer-size 1 --latest-only 1 --drain-grabs 1 --headless 1 --duration 10\n";
    std::cout << "  ./vision_app --mode live --device /dev/video0 --width 640 --height 480 --fourcc MJPG --fps 180 --buffer-size 1 --latest-only 1 --drain-grabs 1 --tag-family auto --target-id 0 --require-target-id 1 --manual-lock-only 1 --warp-soft-max 700 --preview-soft-max 500\n";
    std::cout << "  ./vision_app --mode deploy --device /dev/video0 --width 640 --height 480 --fourcc MJPG --fps 180 --load-warp ../report/warp_package.yml.gz --load-rois ../report/rois.yml\n";
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
        else if (a == "--preview-soft-max") opt.preview_soft_max = std::stoi(need("--preview-soft-max"));
        else if (a == "--temp-preview-square") opt.temp_preview_square = std::stoi(need("--temp-preview-square"));
        else if (a == "--temp-preview-stride") opt.temp_preview_stride = std::stoi(need("--temp-preview-stride"));
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
        if (!bench_capture(opt.device, opt.width, opt.height, opt.fps, opt.fourcc, opt.buffer_size, opt.latest_only, opt.drain_grabs, opt.headless, opt.duration, opt.camera_soft_max, opt.preview_soft_max, stats, err)) {
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
