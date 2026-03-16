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
        else if (key == "manual_lock_only") tag.manual_lock_only = parse_bool(val);
        else if (key == "lock_frames") tag.lock_frames = std::stoi(val);
        else if (key == "max_center_jitter_px") tag.max_center_jitter_px = std::stod(val);
        else if (key == "max_corner_jitter_px") tag.max_corner_jitter_px = std::stod(val);
        else if (key == "min_quad_area_ratio") tag.min_quad_area_ratio = std::stod(val);
        else if (key == "warp_width") dep.warp_width = std::stoi(val);
        else if (key == "warp_height") dep.warp_height = std::stoi(val);
        else if (key == "warp_view_max_side") dep.warp_view_max_side = std::stoi(val);
        else if (key == "interactive_max_side") dep.interactive_max_side = std::stoi(val);
        else if (key == "unsafe_big_frame") dep.unsafe_big_frame = parse_bool(val);
        else if (key == "save_probe_csv") dep.save_probe_csv = val;
        else if (key == "save_test_csv") dep.save_test_csv = val;
        else if (key == "save_report_md") dep.save_report_md = val;
        else if (key == "save_h_path") dep.save_h_path = val;
        else if (key == "save_rois_path") dep.save_rois_path = val;
        else if (key == "save_remap_path") dep.save_remap_path = val;
        else if (key == "load_h_path") dep.load_h_path = val;
        else if (key == "load_rois_path") dep.load_rois_path = val;
        else if (key == "load_remap_path") dep.load_remap_path = val;
        else if (key == "auto_load_h") dep.auto_load_h = parse_bool(val);
        else if (key == "auto_load_rois") dep.auto_load_rois = parse_bool(val);
        else if (key == "auto_load_remap") dep.auto_load_remap = parse_bool(val);
        else if (key == "auto_save_lock") dep.auto_save_lock = parse_bool(val);
        else if (key == "live_preview_raw") dep.live_preview_raw = parse_bool(val);
        else if (key == "live_preview_warp") dep.live_preview_warp = parse_bool(val);
        else if (key == "show_help_overlay") dep.show_help_overlay = parse_bool(val);
        else if (key == "show_status_overlay") dep.show_status_overlay = parse_bool(val);
        else if (key == "use_remap_cache") dep.use_remap_cache = parse_bool(val);
        else if (key == "fixed_point_remap") dep.fixed_point_remap = parse_bool(val);
        else if (key == "save_remap_cache") dep.save_remap_cache = parse_bool(val);
        else if (key == "save_snapshots") dep.save_snapshots = parse_bool(val);
        else if (key == "snapshot_dir") dep.snapshot_dir = val;
        else if (key == "move_step") dep.move_step = std::stod(val);
        else if (key == "size_step") dep.size_step = std::stod(val);
        else if (key == "red_mean_threshold") dep.red_mean_threshold = std::stod(val);
        else if (key == "red_dominance_threshold") dep.red_dominance_threshold = std::stod(val);
        else if (key == "trigger_cooldown_frames") dep.trigger_cooldown_frames = std::stoi(val);
        else if (key == "save_triggered_image_roi") dep.save_triggered_image_roi = parse_bool(val);
        else if (key == "red_roi") parse_roi_ratio_arg(val, rois.red_roi);
        else if (key == "image_roi") parse_roi_ratio_arg(val, rois.image_roi);
    }
    return true;
}

static void print_general_help() {
    std::cout
        << "vision_app (5-module layout: camera / calibrate / stats / deploy / main)\n\n"
        << "Accepted modes:\n"
        << "  --mode probe\n"
        << "  --mode bench\n"
        << "  --mode live\n"
        << "  --mode deploy\n\n"
        << "General arguments:\n"
        << "  --config PATH                Load config file first\n"
        << "  --help-mode probe|bench|live|deploy\n"
        << "  --tag-family auto|16|25|36  AprilTag family search mode\n"
        << "  --warp-width N              Tag rect width used when rectifying the tag quad\n"
        << "  --warp-height N             Tag rect height used when rectifying the tag quad\n"
        << "  --warp-view-max-side N      Max long side for the final full warped preview canvas\n"
        << "  --interactive-max-side N    Safety guard for live/deploy requested camera size\n"
        << "  --unsafe-big-frame 0|1      Override the live/deploy camera safety cap\n"
        << "  --red-mean-threshold X      Gate threshold on mean red in red_roi\n"
        << "  --red-dominance-threshold X Gate threshold on red - max(blue,green)\n\n"
        << "Examples:\n"
        << "  ./vision_app --mode probe\n"
        << "  ./vision_app --help-mode live\n"
        << "  ./vision_app --mode live --tag-family auto --target-id 0\n\n";
}

static void print_probe_help() {
    std::cout
        << "Mode: probe\n"
        << "Purpose:\n"
        << "  Query v4l2-ctl, normalize camera modes, print a compact table, and save probe_table.csv.\n\n"
        << "Most useful arguments:\n"
        << "  --device /dev/video0        Camera node to inspect\n"
        << "  --save-probe-csv PATH       Where to write normalized probe results\n\n"
        << "Example:\n"
        << "  ./vision_app --mode probe --device /dev/video0 --save-probe-csv ../report/probe_table.csv\n\n";
}

static void print_bench_help() {
    std::cout
        << "Mode: bench\n"
        << "Purpose:\n"
        << "  Open one specific camera configuration and measure true capture behavior.\n"
        << "  Use this to choose the mode you will later use for AprilTag locking and warp.\n\n"
        << "Important arguments:\n"
        << "  --device /dev/video0        Camera node\n"
        << "  --width N --height N        Requested capture size\n"
        << "  --fps N                     Requested camera fps\n"
        << "  --fourcc MJPG|YUYV          Requested pixel format\n"
        << "  --buffer-size N             OpenCV/V4L2 requested queue depth; 1 favors freshness\n"
        << "  --latest-only 0|1           When 1, favor the newest frame instead of old queued frames\n"
        << "  --drain-grabs N             Extra grab calls before retrieve; helps drop stale frames\n"
        << "  --warmup N                  Frames to discard right after opening\n"
        << "  --duration N                Benchmark duration in seconds\n"
        << "  --headless 0|1              Disable preview window when benchmarking\n"
        << "  --save-test-csv PATH        Append one result row for this run\n\n"
        << "Good start:\n"
        << "  ./vision_app --mode bench --width 1280 --height 720 --fps 30 --fourcc MJPG --latest-only 1 --drain-grabs 2 --headless 1 --duration 10\n\n";
}

static void print_live_help() {
    std::cout
        << "Mode: live\n"
        << "Purpose:\n"
        << "  Search for an AprilTag, show family/id live, lock the tag, compute the homography from the\n"
        << "  tag quadrilateral itself, then derive a translated/scaled full-view warped preview so the\n"
        << "  entire transformed camera frame is visible in one canvas. After lock, edit ratio-based ROIs\n"
        << "  with keys only. Save H/remap/rois for later deploy.\n\n"
        << "Important camera arguments:\n"
        << "  --device /dev/video0\n"
        << "  --width N --height N --fps N --fourcc MJPG|YUYV\n"
        << "  --latest-only 0|1 --drain-grabs N --buffer-size N\n"
        << "  --interactive-max-side N    Reject larger interactive inputs unless --unsafe-big-frame 1\n\n"
        << "Important tag arguments:\n"
        << "  --tag-family auto|16|25|36  auto tries 36 -> 25 -> 16 and uses the best visible match\n"
        << "  --target-id N               Desired tag id\n"
        << "  --require-target-id 0|1     When 1, ignore tags with different ids\n"
        << "  --manual-lock-only 0|1      When 1, never auto-lock; you press space/enter to lock\n"
        << "  --lock-frames N             Stable frames required before auto-lock\n\n"
        << "Warp and ROI arguments:\n"
        << "  --warp-width N              Target width of the tag rectangle before full-view bbox is computed\n"
        << "  --warp-height N             Target height of the tag rectangle before full-view bbox is computed\n"
        << "  --warp-view-max-side N      Downscale the final warped preview if its long side exceeds this value\n"
        << "  --red-roi x,y,w,h           Ratio-based roi in the final warped preview\n"
        << "  --image-roi x,y,w,h         Ratio-based roi in the final warped preview\n"
        << "  --move-step X --size-step X Keyboard nudge steps for roi editing\n\n"
        << "Deploy-gate arguments already active in live preview:\n"
        << "  --red-mean-threshold X      Trigger when red_roi mean red channel exceeds X\n"
        << "  --red-dominance-threshold X Trigger when red - max(blue,green) exceeds X\n"
        << "  --trigger-cooldown N        Frames to wait after one trigger before allowing the next\n"
        << "  --save-triggered-image-roi  Save image_roi snapshots when gate triggers\n\n"
        << "Key flow:\n"
        << "  raw preview -> tag found -> press space to lock -> full warped preview -> select ROIs -> save\n\n"
        << "Good start:\n"
        << "  ./vision_app --mode live --width 1280 --height 720 --fps 30 --fourcc MJPG --tag-family auto --target-id 0 --manual-lock-only 1 --warp-width 720 --warp-height 720 --warp-view-max-side 900\n\n";
}

static void print_deploy_help() {
    std::cout
        << "Mode: deploy\n"
        << "Purpose:\n"
        << "  Load a saved full-view homography/remap cache and saved ratio ROIs. Warp every new frame into\n"
        << "  the same calibrated warped-preview canvas. Evaluate red_roi first. If the red gate passes,\n"
        << "  extract image_roi and pass it to the built-in infer stub (placeholder for your real model).\n\n"
        << "Important arguments:\n"
        << "  --load-h PATH               Load saved full-view homography json\n"
        << "  --load-rois PATH            Load ratio ROIs\n"
        << "  --load-remap PATH           Load precomputed remap cache for faster warp\n"
        << "  --use-remap-cache 0|1       Use remap instead of warpPerspective at runtime\n"
        << "  --red-mean-threshold X\n"
        << "  --red-dominance-threshold X\n"
        << "  --trigger-cooldown N\n"
        << "  --save-triggered-image-roi  Save image_roi snapshots when gate triggers\n\n"
        << "This version does not yet load a real model. Instead, it runs a lightweight inference stub\n"
        << "on image_roi (mean gray, stddev, edge ratio) so you can verify end-to-end gating and ROI extraction.\n\n"
        << "Good start:\n"
        << "  ./vision_app --mode deploy --load-h ../report/warp_h.json --load-rois ../report/rois.json --load-remap ../report/warp_remap.yml.gz --use-remap-cache 1 --red-mean-threshold 120 --red-dominance-threshold 20\n\n";
}

static void print_help_mode(const std::string& mode) {
    const std::string m = trim(mode);
    if (m == "probe") print_probe_help();
    else if (m == "bench") print_bench_help();
    else if (m == "live") print_live_help();
    else if (m == "deploy") print_deploy_help();
    else {
        print_general_help();
        print_probe_help();
        print_bench_help();
        print_live_help();
        print_deploy_help();
    }
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
                print_help_mode("all");
                return 0;
            } else if (a == "--help-mode") {
                print_help_mode(need("--help-mode"));
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
            else if (a == "--manual-lock-only") tag.manual_lock_only = parse_bool(need("--manual-lock-only"));
            else if (a == "--lock-frames") tag.lock_frames = std::stoi(need("--lock-frames"));
            else if (a == "--max-center-jitter") tag.max_center_jitter_px = std::stod(need("--max-center-jitter"));
            else if (a == "--max-corner-jitter") tag.max_corner_jitter_px = std::stod(need("--max-corner-jitter"));
            else if (a == "--min-quad-area-ratio") tag.min_quad_area_ratio = std::stod(need("--min-quad-area-ratio"));
            else if (a == "--threads") tag.threads = std::stoi(need("--threads"));
            else if (a == "--decimate") tag.decimate = std::stof(need("--decimate"));
            else if (a == "--blur-sigma") tag.blur_sigma = std::stof(need("--blur-sigma"));
            else if (a == "--refine-edges") tag.refine_edges = parse_bool(need("--refine-edges"));
            else if (a == "--warp-width") dep.warp_width = std::stoi(need("--warp-width"));
            else if (a == "--warp-height") dep.warp_height = std::stoi(need("--warp-height"));
            else if (a == "--warp-square") { int n = std::stoi(need("--warp-square")); dep.warp_width = n; dep.warp_height = n; }
            else if (a == "--warp-view-max-side") dep.warp_view_max_side = std::stoi(need("--warp-view-max-side"));
            else if (a == "--interactive-max-side") dep.interactive_max_side = std::stoi(need("--interactive-max-side"));
            else if (a == "--unsafe-big-frame") dep.unsafe_big_frame = parse_bool(need("--unsafe-big-frame"));
            else if (a == "--red-roi") { if (!parse_roi_ratio_arg(need("--red-roi"), rois.red_roi)) throw std::runtime_error("bad --red-roi"); }
            else if (a == "--image-roi") { if (!parse_roi_ratio_arg(need("--image-roi"), rois.image_roi)) throw std::runtime_error("bad --image-roi"); }
            else if (a == "--save-h") dep.save_h_path = need("--save-h");
            else if (a == "--load-h") { dep.load_h_path = need("--load-h"); dep.auto_load_h = true; }
            else if (a == "--save-rois") dep.save_rois_path = need("--save-rois");
            else if (a == "--load-rois") { dep.load_rois_path = need("--load-rois"); dep.auto_load_rois = true; }
            else if (a == "--save-remap") dep.save_remap_path = need("--save-remap");
            else if (a == "--load-remap") { dep.load_remap_path = need("--load-remap"); dep.auto_load_remap = true; }
            else if (a == "--save-probe-csv") dep.save_probe_csv = need("--save-probe-csv");
            else if (a == "--save-test-csv") dep.save_test_csv = need("--save-test-csv");
            else if (a == "--save-report-md") dep.save_report_md = need("--save-report-md");
            else if (a == "--auto-save-lock") dep.auto_save_lock = parse_bool(need("--auto-save-lock"));
            else if (a == "--live-preview-raw") dep.live_preview_raw = parse_bool(need("--live-preview-raw"));
            else if (a == "--live-preview-warp") dep.live_preview_warp = parse_bool(need("--live-preview-warp"));
            else if (a == "--show-help-overlay") dep.show_help_overlay = parse_bool(need("--show-help-overlay"));
            else if (a == "--show-status-overlay") dep.show_status_overlay = parse_bool(need("--show-status-overlay"));
            else if (a == "--use-remap-cache") dep.use_remap_cache = parse_bool(need("--use-remap-cache"));
            else if (a == "--fixed-point-remap") dep.fixed_point_remap = parse_bool(need("--fixed-point-remap"));
            else if (a == "--save-remap-cache") dep.save_remap_cache = parse_bool(need("--save-remap-cache"));
            else if (a == "--save-snapshots") dep.save_snapshots = parse_bool(need("--save-snapshots"));
            else if (a == "--snapshot-dir") dep.snapshot_dir = need("--snapshot-dir");
            else if (a == "--move-step") dep.move_step = std::stod(need("--move-step"));
            else if (a == "--size-step") dep.size_step = std::stod(need("--size-step"));
            else if (a == "--red-mean-threshold") dep.red_mean_threshold = std::stod(need("--red-mean-threshold"));
            else if (a == "--red-dominance-threshold") dep.red_dominance_threshold = std::stod(need("--red-dominance-threshold"));
            else if (a == "--trigger-cooldown") dep.trigger_cooldown_frames = std::stoi(need("--trigger-cooldown"));
            else if (a == "--save-triggered-image-roi") dep.save_triggered_image_roi = parse_bool(need("--save-triggered-image-roi"));
            else throw std::runtime_error("unknown argument: " + a);
        } catch (const std::exception& e) {
            std::cerr << "Argument error: " << e.what() << "\n\n";
            print_help_mode(dep.mode);
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
        if (!write_probe_csv(dep.save_probe_csv, probe)) std::cerr << "Warning: failed to write probe CSV: " << dep.save_probe_csv << "\n";
        else std::cout << "Probe CSV: " << dep.save_probe_csv << "\n";
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
        if (lock.valid) std::cout << "Saved homography: " << dep.save_h_path << "\n";
        std::cout << "Saved ROIs/report: " << dep.save_rois_path << ", " << dep.save_report_md << "\n";
        return 0;
    }

    std::cerr << "Unknown mode: " << dep.mode << "\n";
    print_help_mode("all");
    return 1;
}
