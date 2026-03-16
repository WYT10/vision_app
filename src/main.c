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
        else if (key == "tag_family") tag.family = normalize_tag_family(val);
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
        << "  --mode probe    Enumerate camera formats/resolutions via v4l2-ctl and write\n"
        << "                  a CSV summary. Useful for picking --fourcc / --width / --fps.\n"
        << "  --mode bench    Timed capture benchmark: opens the camera, grabs frames for\n"
        << "                  --duration seconds, then prints FPS / latency statistics and\n"
        << "                  appends one row to the test CSV. No AprilTag detection runs.\n"
        << "                  Use --headless 1 for a clean server-side run, or\n"
        << "                  --preview 1 to watch the stream while benchmarking.\n"
        << "  --mode live     Live AprilTag detect -> auto-lock -> stable warp preview ->\n"
        << "                  interactive ROI edit. A warp preview is shown continuously\n"
        << "                  even before the tag locks so you can frame the shot. Once\n"
        << "                  the tag is stable for --lock-frames frames the homography\n"
        << "                  locks. Drag ROI boxes with the mouse or nudge with keys.\n"
        << "  --mode deploy   Load a previously saved homography + ROIs and run the warp\n"
        << "                  directly without any tag detection.\n\n"
        << "Camera arguments:\n"
        << "  --device /dev/video0         V4L2 device path\n"
        << "  --width  1280                Requested frame width  (camera may round)\n"
        << "  --height 720                 Requested frame height (camera may round)\n"
        << "  --fps    30                  Requested capture frame rate\n"
        << "  --fourcc MJPG               Four-character pixel format (MJPG, YUYV, …)\n"
        << "  --buffer-size 1             V4L2 kernel buffer count (1 = minimal latency)\n"
        << "  --latest-only 1             Drain stale frames before decode (recommended)\n"
        << "  --drain-grabs 2             Extra grab() calls to discard buffered frames\n"
        << "  --warmup 8                  Frames to discard at startup before measuring\n"
        << "  --duration 10               Seconds to run bench/live (0 = run until 'q')\n"
        << "  --headless 0                Set 1 to disable all OpenCV windows\n"
        << "  --preview  1                Set 0 to suppress the raw-camera preview window\n\n"
        << "AprilTag arguments:\n"
        << "  --tag-family <family>        Tag family to detect. Supported values:\n"
        << "                                 auto      – try all families, pick largest hit\n"
        << "                                 36 / tag36h11  – Tag36h11 (default, most robust)\n"
        << "                                 25 / tag25h9   – Tag25h9  (smaller payload)\n"
        << "                                 16 / tag16h5   – Tag16h5  (fewest bits, fastest)\n"
        << "                                 36h10 / tag36h10 – Tag36h10 (legacy)\n"
        << "                               Short forms (16, 25, 36) are normalised automatically.\n"
        << "  --target-id 0               Marker ID to track (-1 = any ID, if require-target-id=0)\n"
        << "  --require-target-id 1       Reject detections whose ID != target-id\n"
        << "  --lock-frames 8             Consecutive stable frames needed to confirm a lock\n"
        << "  --max-center-jitter 3.0     Max center-point drift (px) across lock-frames window\n"
        << "  --max-corner-jitter 4.0     Max per-corner drift (px) across lock-frames window\n"
        << "  --min-quad-area-ratio 0.0025 Tag must cover at least this fraction of frame area\n\n"
        << "Warp/ROI/report arguments:\n"
        << "  --warp-width  1280  --warp-height 720\n"
        << "                              Output resolution of the warped (perspective-corrected)\n"
        << "                              image. ROI ratios are expressed in this coordinate space.\n"
        << "  --red-roi   x,y,w,h         Red ROI as normalised ratios in [0,1] (default: 0.05,0.10,0.20,0.20)\n"
        << "  --image-roi x,y,w,h         Image ROI as normalised ratios in [0,1] (default: 0.30,0.10,0.50,0.60)\n"
        << "  --save-h    ../report/warp_h.json    Path to write homography JSON\n"
        << "  --load-h    ../report/warp_h.json    Path to read  homography JSON (enables auto-load)\n"
        << "  --save-rois ../report/rois.json      Path to write ROI JSON\n"
        << "  --load-rois ../report/rois.json      Path to read  ROI JSON (enables auto-load)\n"
        << "  --save-probe-csv  ../report/probe_table.csv\n"
        << "  --save-test-csv   ../report/test_results.csv\n"
        << "  --save-report-md  ../report/latest_report.md\n"
        << "  --save-snapshots 1          Enable warped-frame snapshot capture (key: c)\n"
        << "  --snapshot-dir   ../report  Directory for snapshot PNGs\n"
        << "  --config ../vision_app.conf Load settings from a .conf file (key=value format)\n\n"
        << "Examples:\n"
        << "  # List available camera modes\n"
        << "  ./vision_app --mode probe\n\n"
        << "  # Benchmark at 1080p MJPG for 15 s, no display\n"
        << "  ./vision_app --mode bench --width 1920 --height 1080 --fps 30 --fourcc MJPG \\\n"
        << "               --duration 15 --headless 1\n\n"
        << "  # Live mode: auto-detect any AprilTag family, warp to 1280x720\n"
        << "  ./vision_app --mode live --tag-family auto --warp-width 1280 --warp-height 720\n\n"
        << "  # Live mode: look for Tag25h9 marker ID 3, tighter jitter thresholds\n"
        << "  ./vision_app --mode live --tag-family 25 --target-id 3 \\\n"
        << "               --max-center-jitter 2.0 --max-corner-jitter 3.0\n\n"
        << "  # Deploy (no detection): load saved homography and ROIs\n"
        << "  ./vision_app --mode deploy --load-h ../report/warp_h.json \\\n"
        << "               --load-rois ../report/rois.json\n";
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
            else if (a == "--tag-family") tag.family = normalize_tag_family(need("--tag-family"));
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
