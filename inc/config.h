#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

struct RoiRatio
{
    double x = 0.0;
    double y = 0.0;
    double w = 0.0;
    double h = 0.0;
};

struct CameraConfig
{
    int index = 0;
    int width = 1280;
    int height = 720;
    int fps = 30;
    int backend = cv::CAP_V4L2;
    bool prefer_mjpg = true;
    int buffer_size = 1;
    int warmup_frames = 30;
    int drop_frames_per_read = 0;
    bool flip_horizontal = false;
};

struct ProbeConfig
{
    std::vector<int> camera_indices {0, 1, 2, 3};
    std::vector<int> widths {640, 1280, 1920};
    std::vector<int> heights {480, 720, 1080};
    std::vector<int> fps_values {30, 60};
    std::vector<int> backends {cv::CAP_V4L2, cv::CAP_ANY};
    int warmup_frames = 30;
    int measure_frames = 120;
    std::string report_dir = "reports";
};

struct TagConfig
{
    std::string family = "auto";
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
};

struct CalibrationData
{
    bool valid = false;
    cv::Mat homography; // 3x3 CV_64F
    int warped_width = 0;
    int warped_height = 0;
    RoiRatio red_roi;
    RoiRatio image_roi;
};

struct AppConfig
{
    CameraConfig camera;
    ProbeConfig probe;
    TagConfig tag;
    TriggerConfig trigger;
    RuntimeConfig runtime;
    CalibrationData calibration;
};

bool load_config(const std::string& path, AppConfig& config, std::string* error = nullptr);
bool save_config(const std::string& path, const AppConfig& config, std::string* error = nullptr);
std::string backend_to_string(int backend);
