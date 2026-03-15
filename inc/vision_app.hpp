#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vision_app {

enum class IoMode {
    Read,
    Grab,
};

struct AppOptions {
    std::string device = "/dev/video0";
    int width = 640;
    int height = 480;
    int fps = 30;
    std::string fourcc = "MJPG";
    int duration_sec = 10;
    int warmup_frames = 8;

    bool probe_only = false;
    bool list_only = false;
    bool headless = true;
    bool show_preview = false;
    bool save_csv = true;
    bool save_md = true;

    std::string report_dir = "../report";
    std::string csv_path = "../report/test_results.csv";
    std::string probe_csv_path = "../report/probe_table.csv";
    std::string markdown_path = "../report/latest_report.md";
    std::string config_path = "../vision_app.conf";

    std::string capture_api = "v4l2";
    IoMode io_mode = IoMode::Grab;
    int buffer_size = 1;
    bool latest_only = true;
    int drain_grabs = 3;
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
    uint64_t empty_frames = 0;
    uint64_t stale_grabs_discarded = 0;

    double elapsed_sec = 0.0;
    double fps_avg = 0.0;
    double fps_min = 0.0;
    double fps_max = 0.0;

    double frame_time_avg_ms = 0.0;
    double frame_time_min_ms = 0.0;
    double frame_time_max_ms = 0.0;

    double actual_width = 0.0;
    double actual_height = 0.0;
    double requested_fps = 0.0;
    double target_ratio = 0.0;
    bool target_met = false;

    bool backend_buffer_request_ok = false;
    double backend_buffer_size_after_set = -1.0;
};

bool load_config_file(const std::string& path, AppOptions& opt);
bool parse_args(int argc, char** argv, AppOptions& opt, std::string& err);
void print_help();

bool probe_camera_modes(const std::string& device, ProbeResult& out, std::string& err);
void print_probe_result(const ProbeResult& probe);
bool write_probe_csv(const std::string& path, const ProbeResult& probe);

bool run_camera_test(const AppOptions& opt, RuntimeStats& stats, std::string& err);
void print_runtime_stats(const AppOptions& opt, const RuntimeStats& stats);

bool ensure_report_dirs(const AppOptions& opt, std::string& err);
bool append_test_csv(const std::string& path, const AppOptions& opt, const RuntimeStats& stats);
bool write_markdown_report(const std::string& path, const AppOptions& opt, const ProbeResult& probe, const RuntimeStats& stats);

} // namespace vision_app
