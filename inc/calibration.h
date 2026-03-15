#pragma once
#include "camera.h"
#include "config.h"
#include "homography.h"
#include <opencv2/opencv.hpp>
#include <string>

namespace app {

enum class CalibrationState {
    LivePreview,
    WarpPreviewReady,
    Locked,
    EditRedRoi,
    EditTargetRoi,
    Done,
    Error
};

class CalibrationSession {
public:
    CalibrationSession(AppConfig& cfg, const std::string& config_path);
    int run();

private:
    bool init(std::string* err = nullptr);
    bool updateLiveFrame(std::string* err = nullptr);
    bool updateWarpPreview(std::string* err = nullptr);
    bool lockCurrentSolution(std::string* err = nullptr);
    bool saveProfile(std::string* err = nullptr);
    void drawUi();
    void handleMouse(int event, int x, int y, int flags);
    bool canSave() const;
    void updateRedRoiTelemetry();

    static void mouseThunk(int event, int x, int y, int flags, void* userdata);

private:
    AppConfig& cfg_;
    std::string config_path_;
    CalibrationState state_ = CalibrationState::LivePreview;

    CameraDevice camera_;
    HomographyEngine homography_;

    cv::Mat live_frame_;
    cv::Mat live_frame_display_;
    cv::Mat candidate_preview_;

    TagDetection live_det_;
    cv::Mat live_candidate_H_;
    cv::Size live_candidate_size_;

    cv::Mat locked_frame_;
    TagDetection locked_det_;
    cv::Mat locked_H_;
    cv::Mat locked_warped_;
    cv::Mat locked_warped_display_;

    bool dragging_ = false;
    cv::Point drag_start_;
    cv::Rect active_rect_px_;
    double red_roi_live_value_ = 0.0;
};

int runCalibration(AppConfig& cfg, const std::string& config_path);

} // namespace app
