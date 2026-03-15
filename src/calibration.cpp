#include "calibration.h"
#include <iostream>

namespace app {

CalibrationSession::CalibrationSession(AppConfig& cfg, const std::string& config_path)
    : cfg_(cfg), config_path_(config_path), homography_(cfg.remap) {}

void CalibrationSession::mouseThunk(int event, int x, int y, int flags, void* userdata) {
    if (auto* self = static_cast<CalibrationSession*>(userdata)) self->handleMouse(event, x, y, flags);
}

bool CalibrationSession::init(std::string* err) {
    if (!camera_.open(cfg_.camera, err)) return false;
#if APP_ENABLE_UI
    if (cfg_.debug.enable_ui) {
        cv::namedWindow("raw", cv::WINDOW_NORMAL);
        cv::namedWindow("warp_preview", cv::WINDOW_NORMAL);
        cv::setMouseCallback("warp_preview", mouseThunk, this);
    }
#endif
    return true;
}

bool CalibrationSession::updateLiveFrame(std::string* err) {
    if (!camera_.read(live_frame_, err)) return false;
    live_frame_display_ = live_frame_.clone();
    live_det_ = {};
    homography_.detectTag(live_frame_, live_det_, err);
    if (live_det_.found) {
        std::vector<std::vector<cv::Point2f>> corners{live_det_.corners};
        std::vector<int> ids{live_det_.id};
        cv::aruco::drawDetectedMarkers(live_frame_display_, corners, ids);
    }
    return true;
}

bool CalibrationSession::updateWarpPreview(std::string* err) {
    candidate_preview_.release();
    live_candidate_H_.release();
    live_candidate_size_ = {};
    if (!live_det_.found) {
        if (err) *err = "tag not found";
        return false;
    }
    if (!homography_.calculateHomography(live_det_, live_candidate_H_, err)) return false;
    if (!homography_.computeWarpedSize(live_frame_, live_candidate_H_, live_candidate_size_, err)) return false;
    cv::Mat warped;
    if (!homography_.warpImage(live_frame_, live_candidate_H_, live_candidate_size_, warped, err)) return false;
    candidate_preview_ = HomographyEngine::makePreview255(warped);
    state_ = CalibrationState::WarpPreviewReady;
    return true;
}

bool CalibrationSession::lockCurrentSolution(std::string* err) {
    if (live_frame_.empty() || !live_det_.found || live_candidate_H_.empty() || !HomographyEngine::validateWarpSize(live_candidate_size_)) {
        if (err) *err = "no valid candidate solution to lock";
        return false;
    }
    locked_frame_ = live_frame_.clone();
    locked_det_ = live_det_;
    locked_H_ = live_candidate_H_.clone();
    if (!homography_.warpImage(locked_frame_, locked_H_, live_candidate_size_, locked_warped_, err)) return false;
    locked_warped_display_ = locked_warped_.clone();
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) cfg_.remap.homography_matrix[r * 3 + c] = locked_H_.at<double>(r, c);
    cfg_.remap.transformed_width = locked_warped_.cols;
    cfg_.remap.transformed_height = locked_warped_.rows;
    cfg_.remap.calibrated = true;
    state_ = CalibrationState::Locked;
    return true;
}

void CalibrationSession::updateRedRoiTelemetry() {
    if (locked_warped_.empty()) return;
    if (state_ == CalibrationState::EditRedRoi || state_ == CalibrationState::Locked) {
        cv::Rect r = active_rect_px_;
        if (r.width <= 0 || r.height <= 0) {
            r = HomographyEngine::roiFromRatio(cfg_.roi.red_roi, locked_warped_.size());
        }
        r &= cv::Rect(0, 0, locked_warped_.cols, locked_warped_.rows);
        if (r.width > 0 && r.height > 0) {
            const cv::Scalar mean_bgr = cv::mean(locked_warped_(r));
            red_roi_live_value_ = mean_bgr[2];
        }
    }
}

void CalibrationSession::drawUi() {
#if APP_ENABLE_UI
    if (!cfg_.debug.enable_ui) return;

    if (!live_frame_display_.empty()) {
        drawStatusText(live_frame_display_, std::string("state: ") + (live_det_.found ? "tag found" : "no tag"), 0, live_det_.found ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255));
        drawStatusText(live_frame_display_, "Enter=preview  L=lock  1=red roi  2=target roi  S=save  Esc=quit", 1, {255,255,0});
        cv::imshow("raw", live_frame_display_);
    }

    cv::Mat preview;
    if (!locked_warped_display_.empty()) preview = locked_warped_display_.clone();
    else if (!candidate_preview_.empty()) preview = candidate_preview_.clone();

    if (!preview.empty()) {
        if (!locked_warped_display_.empty()) {
            cv::Rect red = HomographyEngine::roiFromRatio(cfg_.roi.red_roi, locked_warped_display_.size());
            cv::Rect tgt = HomographyEngine::roiFromRatio(cfg_.roi.target_roi, locked_warped_display_.size());
            cv::rectangle(preview, red, {0,0,255}, 2);
            cv::rectangle(preview, tgt, {255,255,0}, 2);
            if (active_rect_px_.width > 0 && active_rect_px_.height > 0) cv::rectangle(preview, active_rect_px_, {0,255,0}, 2);
            updateRedRoiTelemetry();
            drawStatusText(preview, "red mean: " + std::to_string(static_cast<int>(red_roi_live_value_)) + " threshold: " + std::to_string(cfg_.roi.red_threshold), 0, {0,255,255});
        }
        cv::imshow("warp_preview", preview);
    }
#endif
}

void CalibrationSession::handleMouse(int event, int x, int y, int /*flags*/) {
    if (locked_warped_display_.empty()) return;
    if (!(state_ == CalibrationState::EditRedRoi || state_ == CalibrationState::EditTargetRoi)) return;
    if (event == cv::EVENT_LBUTTONDOWN) {
        dragging_ = true;
        drag_start_ = {x, y};
        active_rect_px_ = cv::Rect(x, y, 1, 1);
    } else if (event == cv::EVENT_MOUSEMOVE && dragging_) {
        const int x0 = std::min(drag_start_.x, x);
        const int y0 = std::min(drag_start_.y, y);
        const int w = std::abs(drag_start_.x - x);
        const int h = std::abs(drag_start_.y - y);
        active_rect_px_ = cv::Rect(x0, y0, w, h) & cv::Rect(0, 0, locked_warped_display_.cols, locked_warped_display_.rows);
    } else if (event == cv::EVENT_LBUTTONUP) {
        dragging_ = false;
        active_rect_px_ &= cv::Rect(0, 0, locked_warped_display_.cols, locked_warped_display_.rows);
        if (active_rect_px_.width > 0 && active_rect_px_.height > 0) {
            if (state_ == CalibrationState::EditRedRoi) {
                cfg_.roi.red_roi = HomographyEngine::ratioFromRect(active_rect_px_, locked_warped_display_.size());
            } else if (state_ == CalibrationState::EditTargetRoi) {
                cfg_.roi.target_roi = HomographyEngine::ratioFromRect(active_rect_px_, locked_warped_display_.size());
            }
        }
    }
}

bool CalibrationSession::canSave() const {
    return cfg_.remap.calibrated && cfg_.remap.transformed_width > 0 && cfg_.remap.transformed_height > 0;
}

bool CalibrationSession::saveProfile(std::string* err) {
    if (!canSave()) {
        if (err) *err = "calibration is incomplete";
        return false;
    }
    return ProfileStore::save(config_path_, cfg_, err);
}

int CalibrationSession::run() {
    std::string err;
    if (!init(&err)) {
        std::cerr << "init failed: " << err << std::endl;
        return 1;
    }

    while (true) {
        if (!updateLiveFrame(&err)) {
            std::cerr << "camera read failed: " << err << std::endl;
            return 1;
        }

        if (!cfg_.debug.manual_warp_preview && live_det_.found) {
            std::string preview_err;
            updateWarpPreview(&preview_err);
        }

        drawUi();

#if APP_ENABLE_UI
        const int key = cfg_.debug.enable_ui ? cv::waitKey(1) : -1;
#else
        const int key = -1;
#endif
        if (key == 27) break;
        if (key == 13 || key == 10) {
            std::string preview_err;
            updateWarpPreview(&preview_err);
        } else if (key == 'l' || key == 'L') {
            std::string lock_err;
            if (!lockCurrentSolution(&lock_err)) std::cerr << "lock failed: " << lock_err << std::endl;
        } else if (key == '1' && !locked_warped_.empty()) {
            state_ = CalibrationState::EditRedRoi;
        } else if (key == '2' && !locked_warped_.empty()) {
            state_ = CalibrationState::EditTargetRoi;
        } else if (key == 's' || key == 'S') {
            std::string save_err;
            if (saveProfile(&save_err)) {
                std::cout << "saved profile to " << config_path_ << std::endl;
                state_ = CalibrationState::Done;
                break;
            }
            std::cerr << "save failed: " << save_err << std::endl;
        }
    }

    camera_.close();
#if APP_ENABLE_UI
    if (cfg_.debug.enable_ui) cv::destroyAllWindows();
#endif
    return 0;
}

int runCalibration(AppConfig& cfg, const std::string& config_path) {
    CalibrationSession s(cfg, config_path);
    return s.run();
}

} // namespace app
