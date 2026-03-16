#include <cctype>
#include <fstream>
#include <iostream>
#include <string>

#include "deploy.hpp"

namespace vision_app {

static std::string trim(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

static bool parse_bool(const std::string& s) {
    return s == "1" || s == "true" || s == "TRUE" || s == "yes" || s == "on";
}

static void load_config(const std::string& path, AppOptions& opt) {
    std::ifstream in(path);
    if (!in.is_open()) return;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        std::string k = trim(line.substr(0, pos));
        std::string v = trim(line.substr(pos + 1));
        if (k == "mode") opt.mode = v;
        else if (k == "device") opt.cam.device = v;
        else if (k == "width") opt.cam.width = std::stoi(v);
        else if (k == "height") opt.cam.height = std::stoi(v);
        else if (k == "fps") opt.cam.fps = std::stoi(v);
        else if (k == "fourcc") opt.cam.fourcc = v;
        else if (k == "duration") opt.duration = std::stoi(v);
        else if (k == "buffer_size") opt.cam.buffer_size = std::stoi(v);
        else if (k == "latest_only") opt.cam.latest_only = parse_bool(v);
        else if (k == "drain_grabs") opt.cam.drain_grabs = std::stoi(v);
        else if (k == "headless") opt.cam.headless = parse_bool(v);
        else if (k == "tag_family") opt.tag.family = v;
        else if (k == "target_id") opt.tag.target_id = std::stoi(v);
        else if (k == "require_target_id") opt.tag.require_target_id = parse_bool(v);
        else if (k == "manual_lock_only") opt.tag.manual_lock_only = parse_bool(v);
        else if (k == "lock_frames") opt.tag.lock_frames = std::stoi(v);
        else if (k == "warp_soft_max") opt.warp_soft_max = std::stoi(v);
        else if (k == "preview_soft_max") opt.preview_soft_max = std::stoi(v);
        else if (k == "save_warp") opt.save_warp = v;
        else if (k == "save_rois") opt.save_rois = v;
        else if (k == "load_warp") opt.load_warp = v;
        else if (k == "load_rois") opt.load_rois = v;
    }
}

static void print_help() {
    std::cout << "vision_app\n"
              << "General:\n"
              << "  --mode probe|bench|live|deploy\n"
              << "  --help-mode probe|bench|live|deploy\n"
              << "  --config ../vision_app.conf\n"
              << "Camera:\n"
              << "  --device /dev/video0 --width 1280 --height 720 --fps 30 --fourcc MJPG\n"
              << "  --buffer-size 1 --latest-only 1 --drain-grabs 2 --headless 0\n"
              << "Live/AprilTag:\n"
              << "  --tag-family auto|16|25|36 --target-id 0 --require-target-id 1 --manual-lock-only 1 --lock-frames 8\n"
              << "Warp / stability:\n"
              << "  --warp-soft-max 900 --preview-soft-max 600\n"
              << "Save/load:\n"
              << "  --save-warp ../report/warp_package.yml.gz --save-rois ../report/rois.yml\n"
              << "  --load-warp ../report/warp_package.yml.gz --load-rois ../report/rois.yml\n";
}

static bool parse_args(int argc, char** argv, AppOptions& opt, std::string& help_mode, std::string& err) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { err = std::string("missing value for ") + name; return {}; }
            return argv[++i];
        };
        if (a == "--mode") opt.mode = need("--mode");
        else if (a == "--help" || a == "-h") { print_help(); return false; }
        else if (a == "--help-mode") { help_mode = need("--help-mode"); return false; }
        else if (a == "--config") load_config(need("--config"), opt);
        else if (a == "--device") opt.cam.device = need("--device");
        else if (a == "--width") opt.cam.width = std::stoi(need("--width"));
        else if (a == "--height") opt.cam.height = std::stoi(need("--height"));
        else if (a == "--fps") opt.cam.fps = std::stoi(need("--fps"));
        else if (a == "--fourcc") opt.cam.fourcc = need("--fourcc");
        else if (a == "--duration") opt.duration = std::stoi(need("--duration"));
        else if (a == "--buffer-size") opt.cam.buffer_size = std::stoi(need("--buffer-size"));
        else if (a == "--latest-only") opt.cam.latest_only = parse_bool(need("--latest-only"));
        else if (a == "--drain-grabs") opt.cam.drain_grabs = std::stoi(need("--drain-grabs"));
        else if (a == "--headless") opt.cam.headless = parse_bool(need("--headless"));
        else if (a == "--tag-family") opt.tag.family = need("--tag-family");
        else if (a == "--target-id") opt.tag.target_id = std::stoi(need("--target-id"));
        else if (a == "--require-target-id") opt.tag.require_target_id = parse_bool(need("--require-target-id"));
        else if (a == "--manual-lock-only") opt.tag.manual_lock_only = parse_bool(need("--manual-lock-only"));
        else if (a == "--lock-frames") opt.tag.lock_frames = std::stoi(need("--lock-frames"));
        else if (a == "--warp-soft-max") opt.warp_soft_max = std::stoi(need("--warp-soft-max"));
        else if (a == "--preview-soft-max") opt.preview_soft_max = std::stoi(need("--preview-soft-max"));
        else if (a == "--save-warp") opt.save_warp = need("--save-warp");
        else if (a == "--save-rois") opt.save_rois = need("--save-rois");
        else if (a == "--load-warp") opt.load_warp = need("--load-warp");
        else if (a == "--load-rois") opt.load_rois = need("--load-rois");
        else { err = "unknown argument: " + a; return false; }
    }
    return true;
}

static void apply_safety_caps(AppOptions& opt) {
    if (opt.cam.width > 1280 || opt.cam.height > 1000) {
        std::cerr << "Warning: camera size too large for stable Pi preview; clamping to 1280x720.\n";
        opt.cam.width = 1280;
        opt.cam.height = 720;
    }
    if (opt.warp_soft_max < 256) opt.warp_soft_max = 256;
    if (opt.warp_soft_max > 1000) opt.warp_soft_max = 1000;
    if (opt.preview_soft_max < 256) opt.preview_soft_max = 256;
    if (opt.preview_soft_max > 800) opt.preview_soft_max = 800;
}

} // namespace vision_app

int main(int argc, char** argv) {
    using namespace vision_app;
    AppOptions opt;
    load_config("../vision_app.conf", opt);
    load_config("./vision_app.conf", opt);
    std::string help_mode, err;
    if (!parse_args(argc, argv, opt, help_mode, err)) {
        if (!help_mode.empty()) print_mode_help(help_mode);
        else if (!err.empty()) std::cerr << err << "\n";
        return err.empty() ? 0 : 1;
    }
    apply_safety_caps(opt);

    if (opt.mode == "probe") {
        CameraProbeResult pr;
        if (!probe_camera(opt.cam.device, pr, err)) {
            std::cerr << err << "\n";
            return 1;
        }
        print_probe(pr);
        return 0;
    }
    if (opt.mode == "bench") {
        if (!run_bench(opt, err)) { std::cerr << err << "\n"; return 1; }
        return 0;
    }
    if (opt.mode == "live") {
        if (!run_live(opt, err)) { std::cerr << err << "\n"; return 1; }
        return 0;
    }
    if (opt.mode == "deploy") {
        if (!run_deploy(opt, err)) { std::cerr << err << "\n"; return 1; }
        return 0;
    }
    std::cerr << "unknown mode: " << opt.mode << "\n";
    return 1;
}
