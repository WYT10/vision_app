#pragma once
#include "config.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace app {

struct CandidateMode {
    int width = 0;
    int height = 0;
    int fps = 0;
    std::string fourcc = "MJPG";
};

struct ProbeRow {
    int requested_width = 0;
    int requested_height = 0;
    int requested_fps = 0;
    std::string requested_fourcc;

    int actual_width = 0;
    int actual_height = 0;
    double actual_fps = 0.0;
    std::string actual_fourcc;

    double measured_loop_fps = 0.0;
    bool open_ok = false;
    bool read_ok = false;
    std::string notes;
};

class CameraDevice {
public:
    CameraDevice() = default;
    ~CameraDevice();

    bool open(const CameraConfig& cfg, std::string* err = nullptr);
    void close();
    bool isOpen() const;

    bool read(cv::Mat& frame, std::string* err = nullptr);
    double measureFps(int warmup_frames, int measure_frames, std::string* err = nullptr);
    void applyConfiguredFlips(cv::Mat& frame) const;

    CameraConfig actualConfig() const;
    const CameraConfig& requestedConfig() const { return requested_; }

private:
    bool applyRequestedMode(const CameraConfig& cfg);
    CameraConfig readBackActualConfig() const;
    static int fourccToInt(const std::string& fourcc);
    static std::string intToFourcc(int v);

private:
    cv::VideoCapture cap_;
    CameraConfig requested_;
    CameraConfig actual_;
    bool open_ = false;
};

class ProbeRunner {
public:
    static bool writeV4L2Report(const AppConfig& cfg, std::string* err = nullptr);
    static bool runOpenCvProbe(const AppConfig& cfg, std::vector<ProbeRow>& rows, std::string* err = nullptr);
    static bool writeCsv(const std::string& path, const std::vector<ProbeRow>& rows, std::string* err = nullptr);
    static bool writeJson(const std::string& path, const AppConfig& cfg, const std::vector<ProbeRow>& rows, std::string* err = nullptr);
};

void drawStatusText(cv::Mat& frame, const std::string& text, int line, const cv::Scalar& color = {0,255,0});
void ensureParentDir(const std::string& path);

} // namespace app
