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

cv::Mat threshold_red_hsv(const cv::Mat& bgr, const RedThresholdConfig& red_cfg) {
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat m1, m2, red_mask;
    cv::inRange(hsv, cv::Scalar(red_cfg.h1_low, red_cfg.s_min, red_cfg.v_min),
                     cv::Scalar(red_cfg.h1_high, 255, 255), m1);
    cv::inRange(hsv, cv::Scalar(red_cfg.h2_low, red_cfg.s_min, red_cfg.v_min),
                     cv::Scalar(red_cfg.h2_high, 255, 255), m2);
    cv::bitwise_or(m1, m2, red_mask);
    int k = std::max(1, red_cfg.morph_k | 1);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {k, k});
    cv::morphologyEx(red_mask, red_mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(red_mask, red_mask, cv::MORPH_CLOSE, kernel);
    return red_mask;
}

struct ZoneStats {
    bool valid = false;
    int red_pixels = 0;
    double red_ratio = 0.0;
    double x_center = -1.0;
};

ZoneStats analyze_zone(const cv::Mat& red_mask, const cv::Mat& valid_mask, const cv::Rect& roi,
                       int zone_min_pixels, double zone_min_ratio) {
    ZoneStats z;
    if (roi.width <= 0 || roi.height <= 0) return z;
    cv::Mat local_red = red_mask(roi).clone();
    cv::Mat local_valid = valid_mask(roi).clone();
    cv::bitwise_and(local_red, local_valid, local_red);
    const int valid_pixels = std::max(1, cv::countNonZero(local_valid));
    z.red_pixels = cv::countNonZero(local_red);
    z.red_ratio = static_cast<double>(z.red_pixels) / static_cast<double>(valid_pixels);
    if (z.red_pixels < zone_min_pixels || z.red_ratio < zone_min_ratio) return z;

    cv::Moments mu = cv::moments(local_red, true);
    if (mu.m00 <= 0.0) return z;
    z.x_center = static_cast<double>(roi.x) + (mu.m10 / mu.m00);
    z.valid = true;
    return z;
}

} // namespace

bool extract_runtime_rois_fixed(const cv::Mat& warped,
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
    out.image_roi_rect = ir;
    out.red_valid_pixels = cv::countNonZero(out.red_mask);
    out.image_valid_pixels = cv::countNonZero(out.image_mask);

    cv::Mat red_mask = threshold_red_hsv(out.red_bgr, red_cfg);
    cv::bitwise_and(red_mask, out.red_mask, red_mask);
    const int valid = std::max(1, out.red_valid_pixels);
    out.red_ratio = static_cast<double>(cv::countNonZero(red_mask)) / static_cast<double>(valid);
    out.red_mask = red_mask;
    out.image_bgr = apply_mask_fill(out.image_bgr, out.image_mask, cv::Scalar(255,255,255));
    err.clear();
    return true;
}

bool extract_runtime_rois_dynamic_stacked(const cv::Mat& warped,
                                          const cv::Mat& valid_mask,
                                          const DynamicRedStackedConfig& dyn_cfg_in,
                                          const RedThresholdConfig& red_cfg,
                                          int stable_counter,
                                          RoiRuntimeData& out,
                                          std::string& err) {
    out = {};
    if (warped.empty() || valid_mask.empty()) {
        err = "empty warped image or mask";
        return false;
    }

    DynamicRedStackedConfig dyn = dyn_cfg_in;
    clamp_dynamic_cfg(dyn, warped.size());
    const int x0 = dyn.search_x0;
    const int x1 = dyn.search_x1 < 0 ? warped.cols : std::min(dyn.search_x1, warped.cols);
    out.upper_zone_rect = cv::Rect(x0, dyn.upper_y0, std::max(1, x1 - x0), std::max(1, dyn.upper_y1 - dyn.upper_y0));
    out.lower_zone_rect = cv::Rect(x0, dyn.lower_y0, std::max(1, x1 - x0), std::max(1, dyn.lower_y1 - dyn.lower_y0));

    const int band_y0 = std::min(out.upper_zone_rect.y, out.lower_zone_rect.y);
    const int band_y1 = std::max(out.upper_zone_rect.y + out.upper_zone_rect.height,
                                 out.lower_zone_rect.y + out.lower_zone_rect.height);
    const cv::Rect band_rect(x0, band_y0, std::max(1, x1 - x0), std::max(1, band_y1 - band_y0));
    out.red_bgr = warped(band_rect).clone();
    out.red_mask = threshold_red_hsv(out.red_bgr, red_cfg);
    cv::Mat band_valid = valid_mask(band_rect).clone();
    cv::bitwise_and(out.red_mask, band_valid, out.red_mask);
    out.red_valid_pixels = cv::countNonZero(band_valid);
    out.red_ratio = static_cast<double>(cv::countNonZero(out.red_mask)) / static_cast<double>(std::max(1, out.red_valid_pixels));

    cv::Mat full_red(warped.rows, warped.cols, CV_8UC1, cv::Scalar(0));
    out.red_mask.copyTo(full_red(band_rect));

    const ZoneStats upper = analyze_zone(full_red, valid_mask, out.upper_zone_rect, dyn.zone_min_pixels, dyn.zone_min_ratio);
    const ZoneStats lower = analyze_zone(full_red, valid_mask, out.lower_zone_rect, dyn.zone_min_pixels, dyn.zone_min_ratio);
    out.upper_valid = upper.valid;
    out.lower_valid = lower.valid;
    out.upper_red_pixels = upper.red_pixels;
    out.lower_red_pixels = lower.red_pixels;
    out.upper_red_ratio = upper.red_ratio;
    out.lower_red_ratio = lower.red_ratio;
    out.x_upper = upper.x_center;
    out.x_lower = lower.x_center;

    const bool x_consistent = upper.valid && lower.valid && std::abs(upper.x_center - lower.x_center) <= dyn.center_x_max_diff;
    if (x_consistent) out.x_center = 0.5 * (upper.x_center + lower.x_center);
    (void)stable_counter;
    out.trigger_ready = x_consistent;

    const int roi_bottom = dyn.upper_y0 - dyn.roi_gap_above_upper_zone;
    const int roi_top = roi_bottom - dyn.roi_height;
    const int roi_left = cvRound((out.x_center >= 0.0 ? out.x_center : 0.5 * warped.cols) - 0.5 * dyn.roi_width);
    const int x_clamped = std::clamp(roi_left, 0, std::max(0, warped.cols - dyn.roi_width));
    const int y_clamped = std::clamp(roi_top, 0, std::max(0, warped.rows - dyn.roi_height));
    out.image_roi_rect = cv::Rect(x_clamped, y_clamped, std::min(dyn.roi_width, warped.cols), std::min(dyn.roi_height, warped.rows));
    out.image_bgr = warped(out.image_roi_rect).clone();
    out.image_mask = valid_mask(out.image_roi_rect).clone();
    out.image_valid_pixels = cv::countNonZero(out.image_mask);
    out.image_bgr = apply_mask_fill(out.image_bgr, out.image_mask, cv::Scalar(255,255,255));
    err.clear();
    return true;
}

} // namespace vision_app
