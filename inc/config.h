#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

struct RoiRatio
{
    double x = 0.0;
    double y = 0.0;
    double w = 0.0;
    double h = 0.0;
};

struct CameraMode
{
    int width = 320;
    int height = 240;
    int fps = 60;
    std::string fourcc = "MJPG";
};

struct CameraConfig
{
    int device_index = 0;
    std::string device_path;
    int backend = cv::CAP_V4L2;
    CameraMode requested_mode;
    int buffer_size = 1;
    int warmup_frames = 20;
    int drop_frames_per_read = 0;
    bool flip_horizontal = false;
    std::vector<CameraMode> probe_candidates {
        {640, 480, 60, "MJPG"},
        {320, 240, 60, "MJPG"},
        {160, 120, 120, "MJPG"},
        {160, 120, 180, "MJPG"}
    };
};

struct TagConfig
{
    std::string family_mode = "auto";
    std::string allowed_family;
    int allowed_id = -1;
    double tag_size_units = 200.0;
    double output_padding_units = 20.0;
    bool lock_on_first_detection = false;
};

struct TriggerConfig
{
    double red_threshold = 180.0;
    double red_margin = 40.0;
    int cooldown_ms = 500;
    bool save_raw = true;
    bool save_warped = true;
    bool save_roi = true;
    std::string capture_dir = "captures";
};

struct RuntimeConfig
{
    bool show_ui = true;
    bool headless_deploy = false;
    int probe_measure_frames = 120;
    std::string report_dir = "reports";
};

struct CalibrationData
{
    bool valid = false;
    CameraMode camera_mode_used;
    cv::Mat homography;
    int warped_width = 0;
    int warped_height = 0;
    RoiRatio red_roi_ratio;
    RoiRatio image_roi_ratio;
};

struct AppConfig
{
    CameraConfig camera;
    TagConfig tag;
    TriggerConfig trigger;
    RuntimeConfig runtime;
    CalibrationData calibration;
};

bool load_config(const std::string& path, AppConfig& config, std::string* error = nullptr);
bool save_config(const std::string& path, const AppConfig& config, std::string* error = nullptr);
std::string backend_to_string(int backend);
int backend_from_string(const std::string& backend_name);
std::string normalize_fourcc_string(const std::string& fourcc);
bool camera_modes_match(const CameraMode& a, const CameraMode& b);
