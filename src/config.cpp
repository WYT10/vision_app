#include "config.hpp"
#include <sstream>

namespace vision_app {

bool load_default_config(AppConfig& cfg, std::string& err) {
    cfg = AppConfig{};
    err.clear();
    return true;
}

bool load_config_ini(const std::string& path, AppConfig& cfg, std::string& err) {
    // Starter blueprint only: leave real INI parsing for the next implementation step.
    (void)path;
    (void)cfg;
    err.clear();
    return true;
}

bool apply_cli_overrides(int argc, char** argv, AppConfig& cfg, std::string& err) {
    // Starter blueprint only: leave real CLI parsing for the next implementation step.
    (void)argc;
    (void)argv;
    (void)cfg;
    err.clear();
    return true;
}

bool load_config_from_argv(int argc, char** argv, AppConfig& cfg, std::string& err) {
    (void)cfg;
    std::string config_path = "vision_app_v2_stacked.ini";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        }
    }
    if (!load_config_ini(config_path, cfg, err)) return false;
    if (!apply_cli_overrides(argc, argv, cfg, err)) return false;
    return true;
}

std::string dump_effective_config(const AppConfig& cfg) {
    std::ostringstream oss;
    oss << "=== effective config ===\n";
    oss << "mode=" << (cfg.mode == AppMode::Probe ? "probe" :
                       cfg.mode == AppMode::Calibrate ? "calibrate" : "deploy") << "\n";
    oss << "roi_mode=" << (cfg.roi_mode == RoiMode::Fixed ? "fixed" : "dynamic_red_stacked") << "\n";
    oss << "camera=" << cfg.camera.device << " "
        << cfg.camera.width << "x" << cfg.camera.height
        << " fps=" << cfg.camera.fps << " fourcc=" << cfg.camera.fourcc << "\n";
    oss << "warp=" << cfg.warp.warp_width << "x" << cfg.warp.warp_height
        << " tag_px=" << cfg.warp.target_tag_px
        << " center=(" << cfg.warp.center_x_ratio << "," << cfg.warp.center_y_ratio << ")\n";
    oss << "upper_zone=[x:" << cfg.dynamic_roi.search_x0 << ":" << cfg.dynamic_roi.search_x1
        << ", y:" << cfg.dynamic_roi.upper_y0 << ":" << cfg.dynamic_roi.upper_y1 << "]\n";
    oss << "lower_zone=[x:" << cfg.dynamic_roi.search_x0 << ":" << cfg.dynamic_roi.search_x1
        << ", y:" << cfg.dynamic_roi.lower_y0 << ":" << cfg.dynamic_roi.lower_y1 << "]\n";
    oss << "dynamic_roi=(w=" << cfg.dynamic_roi.roi_width
        << ", h=" << cfg.dynamic_roi.roi_height
        << ", gap=" << cfg.dynamic_roi.roi_gap_above_upper_zone << ")\n";
    return oss.str();
}

} // namespace vision_app
