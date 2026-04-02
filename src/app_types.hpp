
#pragma once

#include <string>

#include "calibrate.hpp"
#include "model.hpp"
#include "trigger.hpp"

namespace vision_app {

struct AppOptions {
    std::string mode = "probe"; // probe | live | calibrate | deploy
    std::string profile_path = "config/profile.conf";

    std::string device = "/dev/video0";
    int width = 160;
    int height = 120;
    int fps = 120;
    std::string fourcc = "MJPG";
    int buffer_size = 1;
    bool latest_only = true;
    int drain_grabs = 1;

    bool mobile_webcam = false;

    bool ui = true;
    int duration = 10;
    bool draw_overlay = true;

    int camera_soft_max = 1000;
    int camera_preview_max = 800;
    int warp_preview_max = 900;
    int status_width = 720;
    bool show_status_window = true;
    std::string text_sink = "split"; // overlay | status_window | terminal | split

    int warp_width = 384;
    int warp_height = 384;
    int target_tag_px = 128;

    std::string tag_family = "auto";
    int target_id = 0;
    bool require_target_id = true;
    bool manual_lock_only = true;
    int lock_frames = 4;

    std::string save_warp = "report/warp_package.yml.gz";
    std::string load_warp = "report/warp_package.yml.gz";
    std::string save_rois = "report/rois.yml";
    std::string load_rois = "report/rois.yml";
    std::string save_report = "report/latest_report.md";

    std::string trigger_mode = "fixed_rect"; // fixed_rect | dynamic_red_stacked
    RoiConfig default_rois;
    RedThresholdConfig red_cfg;
    FixedRectTriggerConfig fixed_cfg;
    DynamicRedStackedConfig dynamic_cfg;

    ModelConfig model_cfg;
    double model_max_hz = 5.0;

    bool run_red = true;
    bool run_image_roi = true;
    bool run_model = true;

    std::string save_image_roi_dir;
    std::string save_red_roi_dir;
    int save_every_n = 0;
};

} // namespace vision_app
