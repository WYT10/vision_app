#pragma once
#include <array>
#include <string>

namespace app {

struct RoiRatio {
    double x = 0.0;
    double y = 0.0;
    double w = 0.0;
    double h = 0.0;
};

struct CameraConfig {
    std::string probe_report_path = "reports/cam0_v4l2.txt";
    int device_index = 0;
    int width = 640;
    int height = 480;
    int fps = 30;
    std::string fourcc = "MJPG";
    int backend = 200; // cv::CAP_V4L2
    int buffer_size = 1;
    bool flip_horizontal = false;
    bool flip_vertical = false;

    int actual_width = 0;
    int actual_height = 0;
    double actual_fps = 0.0;
    std::string actual_fourcc;
};

struct ProbeConfig {
    std::string csv_path = "reports/cam0_probe.csv";
    std::string json_path = "reports/cam0_probe.json";
};

struct RemapConfig {
    std::string tag_family = "36h11";
    std::string tag_orientation = "0123";
    std::array<double, 9> homography_matrix{{1,0,0,0,1,0,0,0,1}};
    int transformed_width = 0;
    int transformed_height = 0;
    bool calibrated = false;
};

struct RoiConfig {
    RoiRatio red_roi;
    int red_threshold = 180;
    int red_margin = 40;
    int cooldown_ms = 500;
    RoiRatio target_roi;
};

struct ModelConfig {
    std::string backend = "none";
    std::string model_path = "models/model.onnx";
    double confidence_threshold = 0.25;
};

struct DebugConfig {
    bool enable_ui = true;
    bool manual_warp_preview = true;
    bool save_captures = false;
    bool verbose_log = true;
};

struct AppConfig {
    std::string profile_name = "default";
    CameraConfig camera;
    ProbeConfig probe;
    RemapConfig remap;
    RoiConfig roi;
    ModelConfig model;
    DebugConfig debug;
};

class ProfileStore {
public:
    static AppConfig makeDefault();
    static bool save(const std::string& path, const AppConfig& cfg, std::string* err = nullptr);
    static bool load(const std::string& path, AppConfig& cfg, std::string* err = nullptr);
    static bool validate(const AppConfig& cfg, std::string* err = nullptr);
};

} // namespace app
