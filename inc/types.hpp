#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace app {
namespace fs = std::filesystem;

struct RoiNorm {
    double x = 0.25;
    double y = 0.25;
    double w = 0.25;
    double h = 0.25;
};

struct CameraProfile {
    int index = 0;
    int width = 640;
    int height = 480;
    int fps = 30;
    std::string backend = "ANY";
    bool flip_horizontal = false;
    bool use_mjpg = true;
    int warmup_frames = 12;
};

struct TagSpec {
    std::string mode = "auto";   // auto | family | id
    std::string family = "AprilTag 36h11";
    int id = -1;
};

struct RedThreshold {
    int hue_low_1 = 0;
    int hue_high_1 = 12;
    int hue_low_2 = 170;
    int hue_high_2 = 180;
    int sat_min = 100;
    int val_min = 70;
    double ratio_trigger = 0.18;
    int pixel_mean_r_min = 90;
    int cooldown_frames = 20;
};

struct DeployBehavior {
    bool save_trigger_images = true;
    std::string save_dir = "captures";
    bool draw_debug = true;
    bool show_windows = true;
};

struct CalibrationData {
    bool valid = false;
    std::string family = "AprilTag 36h11";
    int id = -1;
    int source_frame_width = 0;
    int source_frame_height = 0;
    int warp_width = 0;
    int warp_height = 0;
    std::array<double, 9> H{};
    std::array<std::array<float, 2>, 4> tag_corners{};
    RoiNorm red_roi;
    RoiNorm image_roi;
};

struct AppConfig {
    CameraProfile camera;
    TagSpec tag;
    RedThreshold red;
    DeployBehavior deploy;
    CalibrationData calibration;
};

struct Detection {
    std::string family;
    int id = -1;
    std::vector<cv::Point2f> corners;
};

struct ProbeResult {
    int camera_index = -1;
    std::string backend;
    int req_w = 0;
    int req_h = 0;
    int req_fps = 0;
    int act_w = 0;
    int act_h = 0;
    double fps_measured = 0.0;
    bool open_ok = false;
    bool read_ok = false;
    bool stable = false;
    double mean_luma = 0.0;
    double luma_std = 0.0;
    std::string note;
};

struct SelectionState {
    bool active = false;
    bool finished = false;
    bool dragging = false;
    cv::Point start{};
    cv::Point current{};
    RoiNorm* target = nullptr;
    int ref_w = 1;
    int ref_h = 1;
    std::string label;
};

struct CliOptions {
    std::string mode = "help"; // probe | calibrate | deploy
    fs::path config_path = "config/system_config.json";
    std::optional<int> camera_index;
    std::optional<int> width;
    std::optional<int> height;
    std::optional<int> fps;
    std::optional<std::string> tag_family;
    std::optional<int> tag_id;
    std::optional<bool> flip;
};

} // namespace app
