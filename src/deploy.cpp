#include "deploy.h"
#include <filesystem>
#include <iostream>

namespace app {

DeploySession::DeploySession(const AppConfig& cfg) : cfg_(cfg) {}

bool DeploySession::validateConfig(std::string* err) const {
    if (!cfg_.remap.calibrated) {
        if (err) *err = "profile is not calibrated";
        return false;
    }
    if (cfg_.remap.transformed_width <= 0 || cfg_.remap.transformed_height <= 0) {
        if (err) *err = "invalid saved transformed size";
        return false;
    }
    return true;
}

bool DeploySession::init(std::string* err) {
    if (!validateConfig(err)) return false;
    if (!camera_.open(cfg_.camera, err)) return false;
    H_ = cv::Mat(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) H_.at<double>(r, c) = cfg_.remap.homography_matrix[r * 3 + c];
    red_roi_px_ = HomographyEngine::roiFromRatio(cfg_.roi.red_roi, {cfg_.remap.transformed_width, cfg_.remap.transformed_height});
    target_roi_px_ = HomographyEngine::roiFromRatio(cfg_.roi.target_roi, {cfg_.remap.transformed_width, cfg_.remap.transformed_height});
#if APP_ENABLE_UI
    if (cfg_.debug.enable_ui) {
        cv::namedWindow("deploy", cv::WINDOW_NORMAL);
        cv::namedWindow("deploy_warped", cv::WINDOW_NORMAL);
    }
#endif
    return true;
}

bool DeploySession::redTriggered(const cv::Mat& warped, double* red_mean_out) const {
    const cv::Rect safe = red_roi_px_ & cv::Rect(0, 0, warped.cols, warped.rows);
    if (safe.width <= 0 || safe.height <= 0) return false;
    const cv::Scalar mean_bgr = cv::mean(warped(safe));
    if (red_mean_out) *red_mean_out = mean_bgr[2];
    return mean_bgr[2] > cfg_.roi.red_threshold && mean_bgr[2] > mean_bgr[1] + cfg_.roi.red_margin && mean_bgr[2] > mean_bgr[0] + cfg_.roi.red_margin;
}

bool DeploySession::forwardInferModel(const cv::Mat& roi) const {
    (void)roi;
    return true;
}

bool DeploySession::processFrame(std::string* err) {
    if (!camera_.read(frame_, err)) return false;
    cv::warpPerspective(frame_, warped_, H_, {cfg_.remap.transformed_width, cfg_.remap.transformed_height});
    if (warped_.empty()) {
        if (err) *err = "warp failed";
        return false;
    }

    double red_mean = 0.0;
    const bool fired = redTriggered(warped_, &red_mean);
    const int64_t now = cv::getTickCount();
    const double elapsed_ms = (now - last_trigger_tick_) * 1000.0 / cv::getTickFrequency();
    if (fired && elapsed_ms >= cfg_.roi.cooldown_ms) {
        last_trigger_tick_ = now;
        const cv::Rect safe = target_roi_px_ & cv::Rect(0, 0, warped_.cols, warped_.rows);
        if (safe.width > 0 && safe.height > 0) {
            cv::Mat crop = warped_(safe).clone();
            forwardInferModel(crop);
            if (cfg_.debug.save_captures) {
                std::filesystem::create_directories("captures");
                cv::imwrite("captures/trigger_raw.png", frame_);
                cv::imwrite("captures/trigger_warped.png", warped_);
                cv::imwrite("captures/trigger_crop.png", crop);
            }
        }
    }

#if APP_ENABLE_UI
    if (cfg_.debug.enable_ui) {
        cv::Mat frame_show = frame_.clone();
        cv::Mat warped_show = warped_.clone();
        cv::rectangle(warped_show, red_roi_px_, {0,0,255}, 2);
        cv::rectangle(warped_show, target_roi_px_, {255,255,0}, 2);
        drawStatusText(warped_show, "red mean: " + std::to_string(static_cast<int>(red_mean)), 0, {0,255,255});
        drawStatusText(warped_show, fired ? "trigger: ON" : "trigger: off", 1, fired ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255));
        cv::imshow("deploy", frame_show);
        cv::imshow("deploy_warped", warped_show);
    }
#endif
    return true;
}

int DeploySession::run() {
    std::string err;
    if (!init(&err)) {
        std::cerr << "deploy init failed: " << err << std::endl;
        return 1;
    }
    while (true) {
        if (!processFrame(&err)) {
            std::cerr << "deploy frame failed: " << err << std::endl;
            break;
        }
#if APP_ENABLE_UI
        const int key = cfg_.debug.enable_ui ? cv::waitKey(1) : -1;
#else
        const int key = -1;
#endif
        if (key == 27) break;
    }
    camera_.close();
#if APP_ENABLE_UI
    if (cfg_.debug.enable_ui) cv::destroyAllWindows();
#endif
    return 0;
}

int runDeploy(const AppConfig& cfg) {
    DeploySession s(cfg);
    return s.run();
}

} // namespace app
