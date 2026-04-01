#include "model.hpp"
#include "deploy.hpp"

#include <cmath>
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

cv::Mat threshold_red_hsv(const cv::Mat& bgr,
                          const cv::Mat& valid_mask,
                          const RedThresholdConfig& red_cfg) {
    if (bgr.empty()) return {};
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
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
    if (!valid_mask.empty()) cv::bitwise_and(red_mask, valid_mask, red_mask);
    return red_mask;
}

cv::Mat build_red_mask_vis(const cv::Mat& red_bgr, const cv::Mat& red_mask) {
    if (red_bgr.empty()) return {};
    cv::Mat vis = red_bgr.clone();
    if (!red_mask.empty()) {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(red_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(vis, contours, -1, cv::Scalar(0, 255, 0), 2);
    }
    return vis;
}

struct BlobHit {
    bool found = false;
    double area = 0.0;
    double center_x = -1.0; // local x inside full mask
};

struct ZoneStats {
    int valid_pixels = 0;
    int red_pixels = 0;
    double red_ratio = 0.0;
};

ZoneStats compute_zone_stats(const cv::Mat& red_mask,
                             const cv::Mat& valid_mask,
                             const cv::Rect& zone_local) {
    ZoneStats s;
    if (zone_local.width <= 0 || zone_local.height <= 0) return s;
    const cv::Rect bounded = zone_local & cv::Rect(0, 0, red_mask.cols, red_mask.rows);
    if (bounded.width <= 0 || bounded.height <= 0) return s;
    if (!valid_mask.empty()) s.valid_pixels = cv::countNonZero(valid_mask(bounded));
    s.red_pixels = cv::countNonZero(red_mask(bounded));
    const int denom = std::max(1, s.valid_pixels);
    s.red_ratio = static_cast<double>(s.red_pixels) / static_cast<double>(denom);
    return s;
}

bool zone_passes_thresholds(const BlobHit& hit,
                            const ZoneStats& stats,
                            const DynamicRedRoiConfig& cfg) {
    if (!hit.found) return false;
    if (stats.red_pixels < std::max(0, cfg.zone_min_pixels)) return false;
    if (stats.red_ratio + 1e-12 < std::max(0.0, cfg.zone_min_ratio)) return false;
    return true;
}

bool band_passes_thresholds(int red_pixels,
                            int valid_pixels,
                            const DynamicRedRoiConfig& cfg) {
    if (red_pixels < std::max(0, cfg.band_min_pixels)) return false;
    const double ratio = static_cast<double>(red_pixels) / static_cast<double>(std::max(1, valid_pixels));
    if (ratio + 1e-12 < std::max(0.0, cfg.band_min_ratio)) return false;
    return true;
}

BlobHit find_best_blob_center_x(const cv::Mat& mask,
                                const cv::Rect& zone_local,
                                int min_area,
                                int max_area) {
    BlobHit hit;
    if (mask.empty() || zone_local.width <= 0 || zone_local.height <= 0) return hit;
    const cv::Rect bounded = zone_local & cv::Rect(0, 0, mask.cols, mask.rows);
    if (bounded.width <= 0 || bounded.height <= 0) return hit;

    cv::Mat zone = mask(bounded).clone();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(zone, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < static_cast<double>(std::max(1, min_area))) continue;
        if (max_area > 0 && area > static_cast<double>(max_area)) continue;
        const cv::Moments mu = cv::moments(contour);
        if (std::abs(mu.m00) < 1e-6) continue;
        const double cx_local_zone = mu.m10 / mu.m00;
        const double cx_local_full = static_cast<double>(bounded.x) + cx_local_zone;
        if (!hit.found || area > hit.area) {
            hit.found = true;
            hit.area = area;
            hit.center_x = cx_local_full;
        }
    }
    return hit;
}

} // namespace

bool extract_runtime_rois(const cv::Mat& warped,
                          const cv::Mat& valid_mask,
                          const std::string& roi_mode,
                          const RoiConfig& rois,
                          const RedThresholdConfig& red_cfg,
                          const DynamicRedRoiConfig& dynamic_cfg,
                          DynamicRedRoiState& dynamic_state,
                          RoiRuntimeData& out,
                          std::string& err) {
    out = {};
    if (warped.empty() || valid_mask.empty()) {
        err = "empty warped image or mask";
        return false;
    }

    const std::string use_mode = normalize_roi_mode(roi_mode);
    out.runtime_mode = use_mode;

    if (use_mode == "fixed") {
        const cv::Rect rr = roi_to_rect(rois.red_roi, warped.size());
        const cv::Rect ir = roi_to_rect(rois.image_roi, warped.size());
        out.fixed_red_rect = rr;
        out.fixed_image_rect = ir;
        out.red_bgr = warped(rr).clone();
        out.image_bgr = warped(ir).clone();
        out.image_mask = valid_mask(ir).clone();
        out.image_valid_pixels = cv::countNonZero(out.image_mask);
        out.trigger_ready = !out.image_bgr.empty();

        const cv::Mat red_valid = valid_mask(rr).clone();
        out.red_valid_pixels = cv::countNonZero(red_valid);
        out.red_mask = threshold_red_hsv(out.red_bgr, red_valid, red_cfg);
        const int valid = std::max(1, out.red_valid_pixels);
        out.red_ratio = static_cast<double>(cv::countNonZero(out.red_mask)) / static_cast<double>(valid);
        out.red_mask_vis = build_red_mask_vis(out.red_bgr, out.red_mask);

        out.image_bgr = apply_mask_fill(out.image_bgr, out.image_mask, cv::Scalar(255,255,255));
        err.clear();
        return true;
    }

    const cv::Rect search_rect = dynamic_search_rect(dynamic_cfg, warped.size());
    out.dynamic_search_rect = search_rect;
    out.dynamic_left_zone_rect = dynamic_left_zone_rect(dynamic_cfg, warped.size());
    out.dynamic_right_zone_rect = dynamic_right_zone_rect(dynamic_cfg, warped.size());
    out.red_bgr = warped(search_rect).clone();
    const cv::Mat search_valid = valid_mask(search_rect).clone();
    out.red_valid_pixels = cv::countNonZero(search_valid);
    out.red_mask = threshold_red_hsv(out.red_bgr, search_valid, red_cfg);

    const int morph_k = std::max(1, dynamic_cfg.morph_k);
    if (!out.red_mask.empty() && morph_k > 1) {
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(morph_k, morph_k));
        cv::morphologyEx(out.red_mask, out.red_mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(out.red_mask, out.red_mask, cv::MORPH_CLOSE, kernel);
    }

    out.red_pixels = cv::countNonZero(out.red_mask);
    const int valid = std::max(1, out.red_valid_pixels);
    out.red_ratio = static_cast<double>(out.red_pixels) / static_cast<double>(valid);
    out.red_mask_vis = build_red_mask_vis(out.red_bgr, out.red_mask);

    const cv::Rect left_local(0, 0, out.dynamic_left_zone_rect.width, out.dynamic_left_zone_rect.height);
    const cv::Rect right_local(out.dynamic_right_zone_rect.x - search_rect.x,
                               0,
                               out.dynamic_right_zone_rect.width,
                               out.dynamic_right_zone_rect.height);
    const BlobHit left_hit = find_best_blob_center_x(out.red_mask, left_local, dynamic_cfg.min_area, dynamic_cfg.max_area);
    const BlobHit right_hit = find_best_blob_center_x(out.red_mask, right_local, dynamic_cfg.min_area, dynamic_cfg.max_area);
    const ZoneStats left_stats = compute_zone_stats(out.red_mask, search_valid, left_local);
    const ZoneStats right_stats = compute_zone_stats(out.red_mask, search_valid, right_local);

    out.left_zone_found = left_hit.found;
    out.right_zone_found = right_hit.found;
    out.left_zone_valid_pixels = left_stats.valid_pixels;
    out.right_zone_valid_pixels = right_stats.valid_pixels;
    out.left_zone_red_pixels = left_stats.red_pixels;
    out.right_zone_red_pixels = right_stats.red_pixels;
    out.left_zone_red_ratio = left_stats.red_ratio;
    out.right_zone_red_ratio = right_stats.red_ratio;
    out.left_zone_passed = zone_passes_thresholds(left_hit, left_stats, dynamic_cfg);
    out.right_zone_passed = zone_passes_thresholds(right_hit, right_stats, dynamic_cfg);
    out.band_passed = band_passes_thresholds(out.red_pixels, out.red_valid_pixels, dynamic_cfg);
    out.left_zone_center_x = left_hit.found ? static_cast<int>(std::lround(search_rect.x + left_hit.center_x)) : -1;
    out.right_zone_center_x = right_hit.found ? static_cast<int>(std::lround(search_rect.x + right_hit.center_x)) : -1;
    out.red_blob_area = static_cast<int>(std::round(left_hit.area + right_hit.area));

    if (dynamic_cfg.require_dual_zone) {
        out.dual_zone_triggered = out.left_zone_passed && out.right_zone_passed && out.band_passed;
        out.red_found = out.dual_zone_triggered;
        out.trigger_ready = out.dual_zone_triggered;
        if (!out.dual_zone_triggered) {
            dynamic_state.reset();
            err.clear();
            return true;
        }
        const double combined_center_x = 0.5 * ((search_rect.x + left_hit.center_x) + (search_rect.x + right_hit.center_x));
        out.red_center_x = static_cast<int>(std::lround(combined_center_x));
        dynamic_state.filtered_center_x = combined_center_x;
        dynamic_state.has_last_center = true;
        dynamic_state.miss_count = 0;
    } else {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(out.red_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        double best_area = -1.0;
        double best_cx_local = -1.0;
        for (const auto& contour : contours) {
            const double area = cv::contourArea(contour);
            if (area < static_cast<double>(std::max(1, dynamic_cfg.min_area))) continue;
            if (dynamic_cfg.max_area > 0 && area > static_cast<double>(dynamic_cfg.max_area)) continue;
            const cv::Moments mu = cv::moments(contour);
            if (std::abs(mu.m00) < 1e-6) continue;
            const double cx = mu.m10 / mu.m00;
            if (area > best_area) {
                best_area = area;
                best_cx_local = cx;
            }
        }

        double use_center_x = -1.0;
        if (best_area > 0.0 && best_cx_local >= 0.0) {
            const double detected_center_x = static_cast<double>(search_rect.x) + best_cx_local;
            const double alpha = std::clamp(dynamic_cfg.center_alpha, 0.0, 1.0);
            if (dynamic_state.has_last_center) {
                dynamic_state.filtered_center_x = alpha * detected_center_x + (1.0 - alpha) * dynamic_state.filtered_center_x;
            } else {
                dynamic_state.filtered_center_x = detected_center_x;
            }
            dynamic_state.has_last_center = true;
            dynamic_state.miss_count = 0;
            out.red_found = true;
            out.trigger_ready = out.band_passed;
            out.red_blob_area = static_cast<int>(std::round(best_area));
            use_center_x = dynamic_state.filtered_center_x;
        } else {
            dynamic_state.miss_count += 1;
            if (dynamic_state.has_last_center && dynamic_state.miss_count <= std::max(0, dynamic_cfg.miss_tolerance)) {
                out.used_last_center = true;
                out.trigger_ready = true;
                use_center_x = dynamic_state.filtered_center_x;
            } else {
                out.used_fallback_center = true;
                out.trigger_ready = false;
                const int fallback = (dynamic_cfg.fallback_center_x >= 0)
                    ? dynamic_cfg.fallback_center_x
                    : (search_rect.x + search_rect.width / 2);
                use_center_x = fallback;
                dynamic_state.filtered_center_x = use_center_x;
                dynamic_state.has_last_center = true;
            }
        }
        out.red_center_x = static_cast<int>(std::lround(use_center_x));
    }

    if (!out.trigger_ready || out.red_center_x < 0) {
        err.clear();
        return true;
    }

    out.dynamic_image_rect = dynamic_image_roi_rect(out.red_center_x, dynamic_cfg, warped.size());
    out.image_bgr = warped(out.dynamic_image_rect).clone();
    out.image_mask = valid_mask(out.dynamic_image_rect).clone();
    out.image_valid_pixels = cv::countNonZero(out.image_mask);
    out.image_bgr = apply_mask_fill(out.image_bgr, out.image_mask, cv::Scalar(255,255,255));

    err.clear();
    return true;
}

} // namespace vision_app
