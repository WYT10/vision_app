#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace vision_app {

struct CameraMode {
    std::string pixfmt;
    int width = 0;
    int height = 0;
    std::vector<double> fps;
};

struct CameraProbeResult {
    std::string device;
    std::string card;
    std::string bus;
    std::vector<CameraMode> modes;
};

struct RuntimeStats {
    uint64_t frames = 0;
    double elapsed_sec = 0.0;
    double fps_avg = 0.0;
    double fps_min = 0.0;
    double fps_max = 0.0;
    double frame_time_avg_ms = 0.0;
    double frame_time_min_ms = 0.0;
    double frame_time_max_ms = 0.0;
    int actual_width = 0;
    int actual_height = 0;
};

bool is_stream_url(const std::string& device);
bool is_device_index_string(const std::string& device);

bool run_command(const std::string& cmd, std::string& out);

bool probe_camera(const std::string& device, CameraProbeResult& out, std::string& err);
void print_probe(const CameraProbeResult& p);

int fourcc_from_string(const std::string& s);
void clamp_camera_size(int& w, int& h, int soft_max);

bool open_capture(cv::VideoCapture& cap,
                  const std::string& device,
                  int& width,
                  int& height,
                  int fps,
                  const std::string& fourcc,
                  int buffer_size,
                  int camera_soft_max,
                  std::string& err);

bool grab_latest_frame(cv::VideoCapture& cap,
                       bool latest_only,
                       int drain_grabs,
                       cv::Mat& frame);

cv::Mat downscale_for_preview(const cv::Mat& src, int preview_soft_max);

bool bench_capture(const std::string& device,
                   int width,
                   int height,
                   int fps,
                   const std::string& fourcc,
                   int buffer_size,
                   bool latest_only,
                   int drain_grabs,
                   bool headless,
                   int duration_sec,
                   int camera_soft_max,
                   int preview_soft_max,
                   RuntimeStats& stats,
                   std::string& err);

} // namespace vision_app
