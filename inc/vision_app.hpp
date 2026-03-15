#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace vision_app
{

    struct AppOptions
    {
        std::string device = "/dev/video0";
        int width = 640;
        int height = 480;
        int fps = 30;
        std::string fourcc = "MJPG";

        int duration_sec = 5;
        int warmup_frames = 10;

        bool probe_only = false;
        bool list_only = false;
        bool headless = false;
        bool show_preview = true;
        bool save_csv = false;

        std::string csv_path = "./report/camera_test.csv";
        std::string config_path = "./vision_app.conf";
    };

    struct CameraMode
    {
        uint32_t width = 0;
        uint32_t height = 0;
        std::string pixel_format; // MJPG / YUYV / etc
        std::vector<double> fps_list;
    };

    struct ProbeResult
    {
        std::string device;
        std::string card_name;
        std::string bus_info;
        std::vector<CameraMode> modes;
    };

    struct RuntimeStats
    {
        uint64_t frames = 0;
        double elapsed_sec = 0.0;
        double fps_avg = 0.0;
        double fps_min = 0.0;
        double fps_max = 0.0;
        double frame_time_avg_ms = 0.0;
        double frame_time_min_ms = 0.0;
        double frame_time_max_ms = 0.0;
        double width = 0.0;
        double height = 0.0;
    };

    bool parse_args(int argc, char **argv, AppOptions &opt, std::string &err);
    bool load_config_file(const std::string &path, AppOptions &opt);

    bool probe_camera_modes(const std::string &device, ProbeResult &out, std::string &err);
    void print_probe_result(const ProbeResult &probe);

    bool run_camera_test(const AppOptions &opt, RuntimeStats &stats, std::string &err);
    bool write_stats_csv(const std::string &path, const AppOptions &opt, const RuntimeStats &stats);

    void print_runtime_stats(const AppOptions &opt, const RuntimeStats &stats);
    void print_help();

} // namespace vision_app