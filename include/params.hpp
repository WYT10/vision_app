#pragma once
#include <string>

namespace vision_app {

enum class AppMode { Probe, Calibrate, Deploy };
enum class RoiMode { Fixed, DynamicRedStacked };

struct CameraConfig {
    std::string device = "/dev/video0";
    int width = 160;
    int height = 120;
    int fps = 120;
    std::string fourcc = "MJPG";
    int buffer_size = 1;
    bool latest_only = true;
    int drain_grabs = 1;
};

struct UiConfig {
    bool enabled = true;
    bool draw_overlay = true;
    int camera_preview_max = 800;
    int warp_preview_max = 900;
    bool text_console = true;
    double text_console_font_scale = 0.55;
    int text_console_padding = 10;
    bool red_mask_window = true;
};

struct ProbeConfig {
    std::string task = "list"; // list | live | snap | bench
    std::string snap_path;
    int bench_duration_sec = 10;
    std::string bench_report_path;
};

struct TagConfig {
    std::string family = "auto";
    int target_id = 0;
    bool require_target_id = true;
    bool manual_lock_only = true;
    int lock_frames = 4;
};

struct WarpConfig {
    int warp_width = 384;
    int warp_height = 384;
    int target_tag_px = 128;
    double center_x_ratio = 0.50;
    double center_y_ratio = 0.42;
};

struct FixedRoiConfig {
    double red_x = 0.08, red_y = 0.08, red_w = 0.18, red_h = 0.18;
    double img_x = 0.32, img_y = 0.10, img_w = 0.50, img_h = 0.55;
};

struct RedThresholdConfig {
    int h1_low = 0, h1_high = 10;
    int h2_low = 170, h2_high = 180;
    int s_min = 80;
    int v_min = 60;

    int morph_open_k = 3;
    int morph_close_k = 3;

    int zone_min_pixels = 24;
    double zone_min_ratio = 0.015;
    int zone_min_blob_area = 20;
    int zone_max_blob_area = 5000;

    int band_min_pixels = 0;
    double band_min_ratio = 0.0;

    int center_x_max_diff = 24;
    int trigger_consecutive_frames = 2;
};

struct DynamicStackedConfig {
    int search_x0 = 0;
    int search_x1 = -1; // -1 means full width

    int upper_y0 = 110;
    int upper_y1 = 135;
    int lower_y0 = 145;
    int lower_y1 = 175;

    int roi_width = 96;
    int roi_height = 96;
    int roi_gap_above_upper_zone = 0;

    double x_smoothing_alpha = 0.70;
    int miss_tolerance_frames = 5;
};

struct AppConfig {
    AppMode mode = AppMode::Calibrate;
    RoiMode roi_mode = RoiMode::DynamicRedStacked;
    CameraConfig camera;
    UiConfig ui;
    ProbeConfig probe;
    TagConfig tag;
    WarpConfig warp;
    FixedRoiConfig fixed_roi;
    RedThresholdConfig red;
    DynamicStackedConfig dynamic_roi;
};

} // namespace vision_app
