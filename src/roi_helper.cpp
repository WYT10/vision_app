
#include "model.hpp"
#include "deploy.hpp"

#include <opencv2/imgproc.hpp>

namespace vision_app {
namespace {

cv::Mat apply_mask_fill(const cv::Mat& img, const cv::Mat& mask, const cv::Scalar& fill) {
    cv::Mat out = img.clone();
    if (out.empty() || mask.empty()) return out;
    cv::Mat inv;
    cv::bitwise_not(mask, inv);
    out.setTo(fill, inv);
    return out;
}

} // namespace

bool extract_runtime_rois(const cv::Mat& warped,
                          const cv::Mat& valid_mask,
                          const RoiConfig& rois,
                          const RedThresholdConfig& red_cfg,
                          RoiRuntimeData& out,
                          std::string& err) {
    out = {};
    if (warped.empty() || valid_mask.empty()) {
        err = "empty warped image or mask";
        return false;
    }
    const cv::Rect rr = roi_to_rect(rois.red_roi, warped.size());
    const cv::Rect ir = roi_to_rect(rois.image_roi, warped.size());
    out.red_bgr = warped(rr).clone();
    out.red_mask = valid_mask(rr).clone();
    out.image_bgr = warped(ir).clone();
    out.image_mask = valid_mask(ir).clone();
    out.red_valid_pixels = cv::countNonZero(out.red_mask);
    out.image_valid_pixels = cv::countNonZero(out.image_mask);

    cv::Mat hsv;
    cv::cvtColor(out.red_bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat m1, m2, red_mask;
    cv::inRange(hsv,
                cv::Scalar(red_cfg.h1_low, red_cfg.s_min, red_cfg.v_min),
                cv::Scalar(red_cfg.h1_high, 255, 255),
                m1);
    cv::inRange(hsv,
                cv::Scalar(red_cfg.h2_low, red_cfg.s_min, red_cfg.v_min),
                cv::Scalar(red_cfg.h2_high, 255, 255),
                m2);
    cv::bitwise_or(m1, m2, red_mask);
    cv::bitwise_and(red_mask, out.red_mask, red_mask);
    const int valid = std::max(1, out.red_valid_pixels);
    out.red_ratio = static_cast<double>(cv::countNonZero(red_mask)) / static_cast<double>(valid);

    // white-fill invalid pixels in image roi for model/debug saving consistency
    out.image_bgr = apply_mask_fill(out.image_bgr, out.image_mask, cv::Scalar(255,255,255));
    err.clear();
    return true;
}

} // namespace vision_app
