#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vision_app {

struct AppOptions {
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

    bool save_csv = true;
    bool save_probe_csv = true;
    bool save_md_report = true;

    std::string report_dir = "../report";
    std::string csv_path = "../report/test_results.csv";
    std::string probe_csv_path = "../report/probe_table.csv";
    std::string md_report_path = "../report/latest_report.md";
    std::string config_path = "./vision_app.conf";
};

struct CameraMode {
    std::string pixel_format;
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<double> fps_list;
};

struct ProbeResult {
    std::string device;
    std::string card_name;
    std::string bus_info;
    std::vector<CameraMode> modes;
};

struct RuntimeStats {
    uint64_t frames = 0;
    uint64_t read_failures = 0;
    double elapsed_sec = 0.0;

    double fps_avg = 0.0;
    double fps_min = 0.0;
    double fps_max = 0.0;

    double frame_time_avg_ms = 0.0;
    double frame_time_min_ms = 0.0;
    double frame_time_max_ms = 0.0;
    double frame_time_stddev_ms = 0.0;

    double width = 0.0;
    double height = 0.0;
    double target_ratio = 0.0;
    bool target_met = false;
    bool mode_reported = false;
    std::string stability = "unknown";
};

bool parse_args(int argc, char** argv, AppOptions& opt, std::string& err);
bool load_config_file(const std::string& path, AppOptions& opt);
void print_help();

bool probe_camera_modes(const std::string& device, ProbeResult& out, std::string& err);
void print_probe_result(const ProbeResult& probe);
bool probe_has_mode(const ProbeResult& probe, const AppOptions& opt);
bool write_probe_csv(const std::string& path, const ProbeResult& probe);

bool run_camera_test(const AppOptions& opt, const ProbeResult& probe, RuntimeStats& stats, std::string& err);
void print_runtime_stats(const AppOptions& opt, const RuntimeStats& stats);
bool write_stats_csv(const std::string& path, const AppOptions& opt, const RuntimeStats& stats);
bool write_markdown_report(const std::string& path, const AppOptions& opt, const ProbeResult& probe, const RuntimeStats& stats);

bool ensure_parent_dir(const std::string& path);

} // namespace vision_app
