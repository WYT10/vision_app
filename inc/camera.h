#pragma once

#include "config.h"
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

struct ProbeResult
{
    int camera_index = -1;
    int backend = cv::CAP_ANY;
    int requested_width = 0;
    int requested_height = 0;
    int requested_fps = 0;
    int actual_width = 0;
    int actual_height = 0;
    double actual_fps_property = 0.0;
    double measured_fps = 0.0;
    bool opened = false;
    bool stable = false;
    double mean_luma = 0.0;
    double stddev_luma = 0.0;
    std::string note;
};

bool open_camera(cv::VideoCapture& cap, const CameraConfig& cfg, std::string* error = nullptr);
bool read_frame(cv::VideoCapture& cap, cv::Mat& frame, int drop_frames_per_read = 0);
std::vector<ProbeResult> run_camera_probe(const AppConfig& cfg);
bool write_probe_report(const std::string& report_dir, const std::vector<ProbeResult>& results, std::string* json_path = nullptr, std::string* csv_path = nullptr);
