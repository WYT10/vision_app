#pragma once

#include "config.h"
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

struct ProbeResult
{
    int camera_index = -1;
    std::string device_path;
    int backend = cv::CAP_ANY;
    CameraMode requested_mode;
    CameraMode actual_mode;
    bool opened = false;
    bool stable = false;
    double measured_fps = 0.0;
    double mean_luma = 0.0;
    double stddev_luma = 0.0;
    std::string note;
};

bool open_camera(cv::VideoCapture& cap, const CameraConfig& cfg, CameraMode* actual_mode = nullptr, std::string* error = nullptr);
bool read_frame(cv::VideoCapture& cap, cv::Mat& frame, int drop_frames_per_read = 0);
std::vector<ProbeResult> run_camera_probe(const AppConfig& cfg);
bool write_probe_report(const std::string& report_dir, const std::vector<ProbeResult>& results, std::string* json_path = nullptr, std::string* csv_path = nullptr);
