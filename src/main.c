#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "camera.hpp"
#include "calibrate.hpp"
#include "stats.hpp"
#include "deploy.hpp"

namespace {

using namespace vision_app;

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

static bool parse_roi_ratio_arg(const std::string& s, RoiRatio& roi) {
    std::stringstream ss(s);
    std::string tok;
    double vals[4]{};
    int i = 0;
    while (std::getline(ss, tok, ',') && i < 4) vals[i++] = std::stod(trim(tok));
    if (i != 4) return false;
    roi = {vals[0], vals[1], vals[2], vals[3]};
    return clamp_and_validate_roi(roi);
}

static bool load_config_file(const std::string& path,
                             CameraConfig& cam,
                             AprilTagConfig& tag,
                             DeployConfig& dep,
                             RoiSet& rois) {
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        const size_t pos = line.find('=');
        if (pos == std::string::npos) continue;
        const std::string key = trim(line.substr(0, pos));
        const std::string val = trim(line.substr(pos + 1));

        if (key == "mode") dep.mode = val;
        else if (key == "device") cam.device = val;
        else if (key == "width") cam.width = std::stoi(val);
        else if (key == "height") cam.height = std::stoi(val);
        else if (key == "fps") cam.fps = std::stoi(val);
        else if (key == "fourcc") cam.fourcc = val;
        else if (key == "buffer_size") cam.buffer_size = std::stoi(val);
        else if (key == "latest_only") cam.latest_only = parse_bool(val);
        else if (key == "drain_grabs") cam.drain_grabs = std::stoi(val);
        else if (key == "warmup_frames") cam.warmup_frames = std::stoi(val);
        else if (key == "duration_sec") cam.duration_sec = std::stoi(val);
        else if (key == "preview") cam.preview = parse_bool(val);
        else if (key == "headless") cam.headless = parse_bool(val);
        else if (key == "tag_family") tag.family = val;
        else if (key == "target_id") tag.target_id = std::stoi(val);
        else if (key == "require_target_id") tag.require_target_id = parse_bool(val);
        else if (key == "lock_frames") tag.lock_frames = std::stoi(val);
        else if (key == "max_center_jitter_px") tag.max_center_jitter_px = std::stod(val);
        else if (key == "max_corner_jitter_px") tag.max_corner_jitter_px = std::stod(val);
        else if (key == "min_quad_area_ratio") tag.min_quad_area_ratio = std::stod(val);
        else if (key == "warp_width") dep.warp_width = std::stoi(val);
        else if (key == "warp_height") dep.warp_height = std::stoi(val);
        else if (key == "save_probe_csv") dep.save_probe_csv = val;
        else if (key == "save_test_csv") dep.save_test_csv = val;
        else if (key == "save_report_md") dep.save_report_md = val;
        else if (key == "save_h_path") dep.save_h_path = val;
        else if (key == "save_rois_path") dep.save_rois_path = val;
        else if (key == "load_h_path") dep.load_h_path = val;
        else if (key == "load_rois_path") dep.load_rois_path = val;
        else if (key == "auto_load_h") dep.auto_load_h = parse_bool(val);
        else if (key == "auto_load_rois") dep.auto_load_rois = parse_bool(val);
        else if (key == "red_roi") parse_roi_ratio_arg(val, rois.red_roi);
        else if (key == "image_roi") parse_roi_ratio_arg(val, rois.image_roi);
    }
    return true;
}

static void print_help() {
    std::cout
        << "vision_app (5-module layout: camera / calibrate / stats / deploy / main)\n\n"
        << "Modes:\n"
        << "  --mode probe    Probe camera modes and write probe CSV\n"
        << "  --mode bench    Timed capture benchmark only\n"
        << "  --mode live     Live AprilTag detect -> lock -> warp preview -> ROI edit\n"
        << "  --mode deploy   Load saved homography + ROIs and run warp directly\n\n"
        << "Camera arguments:\n"
        << "  --device /dev/video0\n"
        << "  --width 1280 --height 720 --fps 30 --fourcc MJPG\n"
        << "  --buffer-size 1 --latest-only 1 --drain-grabs 2\n"
        << "  --warmup 8 --duration 10\n"
        << "  --headless 0 --preview 1\n\n"
        << "AprilTag arguments:\n"
        << "  --tag-family tag36h11\n"
        << "  --target-id 0 --require-target-id 1\n"
        << "  --lock-frames 8\n"
        << "  --max-center-jitter 3.0 --max-corner-jitter 4.0\n"
        << "  --min-quad-area-ratio 0.0025\n\n"
        << "Warp/ROI/report arguments:\n"
        << "  --warp-width 1280 --warp-height 720\n"
        << "  --red-roi 0.05,0.10,0.20,0.20\n"
        << "  --image-roi 0.30,0.10,0.50,0.60\n"
        << "  --save-h ../report/warp_h.json --load-h ../report/warp_h.json\n"
        << "  --save-rois ../report/rois.json --load-rois ../report/rois.json\n"
        << "  --save-probe-csv ../report/probe_table.csv\n"
        << "  --save-test-csv ../report/test_results.csv\n"
        << "  --save-report-md ../report/latest_report.md\n"
        << "  --config ../vision_app.conf\n\n"
        << "Examples:\n"
        << "  ./vision_app --mode probe\n"
        << "  ./vision_app --mode bench --width 1280 --height 720 --fps 30 --fourcc MJPG --headless 1\n"
        << "  ./vision_app --mode live --tag-family tag36h11 --target-id 0 --warp-width 1280 --warp-height 720\n"
        << "  ./vision_app --mode deploy --load-h ../report/warp_h.json --load-rois ../report/rois.json\n";
}

} // namespace

int main(int argc, char** argv) {
    using namespace vision_app;

    CameraConfig cam;
    AprilTagConfig tag;
    DeployConfig dep;
    RoiSet rois;
    std::string config_path = "../vision_app.conf";

    load_config_file(config_path, cam, tag, dep, rois);

    std::string err;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };

        try {
            if (a == "-h" || a == "--help") {
                print_help();
                return 0;
            } else if (a == "--config") {
                config_path = need("--config");
                load_config_file(config_path, cam, tag, dep, rois);
            } else if (a == "--mode") dep.mode = need("--mode");
            else if (a == "--device") cam.device = need("--device");
            else if (a == "--width") cam.width = std::stoi(need("--width"));
            else if (a == "--height") cam.height = std::stoi(need("--height"));
            else if (a == "--fps") cam.fps = std::stoi(need("--fps"));
            else if (a == "--fourcc") cam.fourcc = need("--fourcc");
            else if (a == "--buffer-size") cam.buffer_size = std::stoi(need("--buffer-size"));
            else if (a == "--latest-only") cam.latest_only = parse_bool(need("--latest-only"));
            else if (a == "--drain-grabs") cam.drain_grabs = std::stoi(need("--drain-grabs"));
            else if (a == "--warmup") cam.warmup_frames = std::stoi(need("--warmup"));
            else if (a == "--duration") cam.duration_sec = std::stoi(need("--duration"));
            else if (a == "--preview") cam.preview = parse_bool(need("--preview"));
            else if (a == "--headless") { cam.headless = parse_bool(need("--headless")); if (cam.headless) cam.preview = false; }
            else if (a == "--tag-family") tag.family = need("--tag-family");
            else if (a == "--target-id") tag.target_id = std::stoi(need("--target-id"));
            else if (a == "--require-target-id") tag.require_target_id = parse_bool(need("--require-target-id"));
            else if (a == "--lock-frames") tag.lock_frames = std::stoi(need("--lock-frames"));
            else if (a == "--max-center-jitter") tag.max_center_jitter_px = std::stod(need("--max-center-jitter"));
            else if (a == "--max-corner-jitter") tag.max_corner_jitter_px = std::stod(need("--max-corner-jitter"));
            else if (a == "--min-quad-area-ratio") tag.min_quad_area_ratio = std::stod(need("--min-quad-area-ratio"));
            else if (a == "--warp-width") dep.warp_width = std::stoi(need("--warp-width"));
            else if (a == "--warp-height") dep.warp_height = std::stoi(need("--warp-height"));
            else if (a == "--red-roi") { if (!parse_roi_ratio_arg(need("--red-roi"), rois.red_roi)) throw std::runtime_error("bad --red-roi"); }
            else if (a == "--image-roi") { if (!parse_roi_ratio_arg(need("--image-roi"), rois.image_roi)) throw std::runtime_error("bad --image-roi"); }
            else if (a == "--save-h") dep.save_h_path = need("--save-h");
            else if (a == "--load-h") { dep.load_h_path = need("--load-h"); dep.auto_load_h = true; }
            else if (a == "--save-rois") dep.save_rois_path = need("--save-rois");
            else if (a == "--load-rois") { dep.load_rois_path = need("--load-rois"); dep.auto_load_rois = true; }
            else if (a == "--save-probe-csv") dep.save_probe_csv = need("--save-probe-csv");
            else if (a == "--save-test-csv") dep.save_test_csv = need("--save-test-csv");
            else if (a == "--save-report-md") dep.save_report_md = need("--save-report-md");
            else if (a == "--save-snapshots") dep.save_snapshots = parse_bool(need("--save-snapshots"));
            else if (a == "--snapshot-dir") dep.snapshot_dir = need("--snapshot-dir");
            else {
                throw std::runtime_error("unknown argument: " + a);
            }
        } catch (const std::exception& e) {
            std::cerr << "Argument error: " << e.what() << "\n\n";
            print_help();
            return 1;
        }
    }

    if (dep.mode == "probe") {
        ProbeResult probe;
        if (!probe_camera_modes(cam.device, probe, err)) {
            std::cerr << "Probe failed: " << err << "\n";
            return 1;
        }
        print_probe_result(probe);
        if (!write_probe_csv(dep.save_probe_csv, probe)) {
            std::cerr << "Warning: failed to write probe CSV: " << dep.save_probe_csv << "\n";
        } else {
            std::cout << "Probe CSV: " << dep.save_probe_csv << "\n";
        }
        return 0;
    }

    if (dep.mode == "bench") {
        CaptureStats capture;
        if (!run_bench_mode(cam, dep, capture, err)) {
            std::cerr << "Bench failed: " << err << "\n";
            return 1;
        }
        return 0;
    }

    if (dep.mode == "live" || dep.mode == "deploy") {
        CaptureStats capture;
        StageStats stage;
        HomographyLock lock;
        if (!run_live_or_deploy_mode(cam, tag, dep, rois, capture, stage, lock, err)) {
            std::cerr << "Run failed: " << err << "\n";
            return 1;
        }
        if (lock.valid) {
            std::cout << "Saved homography: " << dep.save_h_path << "\n";
        }
        std::cout << "Saved ROIs/report: " << dep.save_rois_path << ", " << dep.save_report_md << "\n";
        return 0;
    }

    std::cerr << "Unknown mode: " << dep.mode << "\n";
    print_help();
    return 1;
}
