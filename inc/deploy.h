#pragma once
#include "camera.h"
#include "config.h"
#include "homography.h"
#include <opencv2/opencv.hpp>

namespace app {

class DeploySession {
public:
    explicit DeploySession(const AppConfig& cfg);
    int run();

private:
    bool init(std::string* err = nullptr);
    bool validateConfig(std::string* err = nullptr) const;
    bool processFrame(std::string* err = nullptr);
    bool redTriggered(const cv::Mat& warped, double* red_mean_out = nullptr) const;
    bool forwardInferModel(const cv::Mat& roi) const;

private:
    AppConfig cfg_;
    CameraDevice camera_;
    cv::Mat H_;
    cv::Mat frame_;
    cv::Mat warped_;
    cv::Rect red_roi_px_;
    cv::Rect target_roi_px_;
    int64_t last_trigger_tick_ = 0;
};

int runDeploy(const AppConfig& cfg);

} // namespace app
