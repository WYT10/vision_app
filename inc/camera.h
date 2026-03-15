#pragma once

#include "config.h"
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

/*
==============================================================================
camera.h
==============================================================================
Purpose
    Shared camera open/read/probe layer.

Design split
    - open_camera / read_frame: hot path used by calibration + deploy
    - run_camera_probe        : discovery tool used before configuration

Probe design
    The probe has two layers:
      1. Driver enumeration: capture the same V4L2 information that
         `v4l2-ctl --list-devices` and `--list-formats-ext` show.
      2. Real capture test: request candidate modes through OpenCV, measure
         actual frame delivery, and record what the driver really applied.
==============================================================================
*/

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

struct ProbeReport
{
    std::string v4l2_device;
    std::string list_devices_output;
    std::string list_formats_output;
    std::vector<ProbeResult> results;
};

/*
------------------------------------------------------------------------------
open_camera
------------------------------------------------------------------------------
Input
    cfg         : camera configuration from JSON

Output
    cap         : opened VideoCapture
    actual_mode : actual width/height/fps/fourcc read back from backend

Return
    true if the camera opens and survives warmup reads.
------------------------------------------------------------------------------
*/
bool open_camera(cv::VideoCapture& cap, const CameraConfig& cfg, CameraMode* actual_mode = nullptr, std::string* error = nullptr);

/*
------------------------------------------------------------------------------
read_frame
------------------------------------------------------------------------------
Input
    cap                 : opened VideoCapture
    drop_frames_per_read: optional extra grab() calls to flush old frames

Output
    frame : newest frame available after optional drops
------------------------------------------------------------------------------
*/
bool read_frame(cv::VideoCapture& cap, cv::Mat& frame, int drop_frames_per_read = 0);

/* Probe camera modes and capture both driver enumeration and real test data. */
ProbeReport run_camera_probe(const AppConfig& cfg);

/* Write probe report as JSON + CSV + raw V4L2 text dump when available. */
bool write_probe_report(const std::string& report_dir, const ProbeReport& report, std::string* json_path = nullptr, std::string* csv_path = nullptr, std::string* txt_path = nullptr);
