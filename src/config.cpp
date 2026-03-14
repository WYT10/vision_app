#include "config.hpp"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "utils.hpp"

namespace app {
using json = nlohmann::json;

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(RoiNorm, x, y, w, h)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CameraProfile, index, width, height, fps, backend, flip_horizontal, use_mjpg, warmup_frames)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TagSpec, mode, family, id)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(RedThreshold, hue_low_1, hue_high_1, hue_low_2, hue_high_2, sat_min, val_min, ratio_trigger, pixel_mean_r_min, cooldown_frames)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DeployBehavior, save_trigger_images, save_dir, draw_debug, show_windows)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CalibrationData, valid, family, id, source_frame_width, source_frame_height, warp_width, warp_height, H, tag_corners, red_roi, image_roi)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AppConfig, camera, tag, red, deploy, calibration)

AppConfig defaultConfig() {
    return AppConfig{};
}

AppConfig loadConfig(const fs::path& path) {
    if (!fs::exists(path)) {
        AppConfig config = defaultConfig();
        ensureDir(path.parent_path());
        std::ofstream(path) << json(config).dump(2) << '\n';
        return config;
    }

    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open config: " + path.string());
    }

    json j;
    in >> j;
    return j.get<AppConfig>();
}

void saveConfig(const AppConfig& config, const fs::path& path) {
    ensureDir(path.parent_path());
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write config: " + path.string());
    }
    out << json(config).dump(2) << '\n';
}

} // namespace app
