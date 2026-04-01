#include "red_trigger.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace vision_app {
namespace {

static cv::Rect clamp_rect(const cv::Rect& r, const cv::Size& sz) {
    const int x0 = std::clamp(r.x, 0, std::max(0, sz.width));
    const int y0 = std::clamp(r.y, 0, std::max(0, sz.height));
    const int x1 = std::clamp(r.x + r.width, 0, std::max(0, sz.width));
    const int y1 = std::clamp(r.y + r.height, 0, std::max(0, sz.height));
    return cv::Rect(x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0));
}

static int normalize_search_x1(int x1, int width) {
    return (x1 < 0) ? width : std::clamp(x1, 0, width);
}

static ZoneStats measure_zone(const cv::Mat& red_mask_full,
                              const cv::Mat& valid_mask,
                              const cv::Rect& zone,
                              const RedThresholdConfig& thr) {
    ZoneStats s;
    if (zone.width <= 0 || zone.height <= 0) return s;

    cv::Mat zone_mask = red_mask_full(zone);
    cv::Mat zone_valid = valid_mask(zone);
    s.valid_pixels = cv::countNonZero(zone_valid);
    s.red_pixels = cv::countNonZero(zone_mask);
    if (s.valid_pixels > 0) {
        s.red_ratio = static_cast<double>(s.red_pixels) / static_cast<double>(s.valid_pixels);
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(zone_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double best_area = -1.0;
    cv::Rect best_local;
    for (const auto& c : contours) {
        const double area = cv::contourArea(c);
        if (area < thr.zone_min_blob_area) continue;
        if (area > thr.zone_max_blob_area) continue;
        cv::Rect b = cv::boundingRect(c);
        if (area > best_area) {
            best_area = area;
            best_local = b;
        }
    }

    if (best_area > 0.0) {
        s.best_blob_bbox = best_local + zone.tl();
        s.x_center = static_cast<float>(s.best_blob_bbox.x + 0.5 * s.best_blob_bbox.width);
    }

    s.pass = (s.red_pixels >= thr.zone_min_pixels) &&
             (s.red_ratio >= thr.zone_min_ratio) &&
             (best_area >= thr.zone_min_blob_area);
    return s;
}

} // namespace

bool compute_red_mask_stacked(const cv::Mat& warped,
                              const cv::Mat& valid_mask,
                              const DynamicStackedConfig& dyn,
                              const RedThresholdConfig& thr,
                              StackedRedDebug& out,
                              std::string& err) {
    out = {};
    err.clear();
    if (warped.empty() || valid_mask.empty()) {
        err = "empty warped or valid mask";
        return false;
    }
    if (warped.size() != valid_mask.size()) {
        err = "warped and valid mask size mismatch";
        return false;
    }

    const int x0 = std::clamp(dyn.search_x0, 0, warped.cols);
    const int x1 = normalize_search_x1(dyn.search_x1, warped.cols);
    if (x1 <= x0) {
        err = "invalid search x range";
        return false;
    }

    out.upper_zone = clamp_rect(cv::Rect(x0, dyn.upper_y0, x1 - x0, dyn.upper_y1 - dyn.upper_y0), warped.size());
    out.lower_zone = clamp_rect(cv::Rect(x0, dyn.lower_y0, x1 - x0, dyn.lower_y1 - dyn.lower_y0), warped.size());
    const int band_y0 = std::min(out.upper_zone.y, out.lower_zone.y);
    const int band_y1 = std::max(out.upper_zone.y + out.upper_zone.height,
                                 out.lower_zone.y + out.lower_zone.height);
    out.full_search_rect = clamp_rect(cv::Rect(x0, band_y0, x1 - x0, band_y1 - band_y0), warped.size());

    if (out.upper_zone.width <= 0 || out.upper_zone.height <= 0 ||
        out.lower_zone.width <= 0 || out.lower_zone.height <= 0) {
        err = "upper or lower zone collapsed after clamping";
        return false;
    }

    cv::Mat hsv;
    cv::cvtColor(warped, hsv, cv::COLOR_BGR2HSV);
    cv::Mat m1, m2, red_mask;
    cv::inRange(hsv, cv::Scalar(thr.h1_low, thr.s_min, thr.v_min), cv::Scalar(thr.h1_high, 255, 255), m1);
    cv::inRange(hsv, cv::Scalar(thr.h2_low, thr.s_min, thr.v_min), cv::Scalar(thr.h2_high, 255, 255), m2);
    cv::bitwise_or(m1, m2, red_mask);
    cv::bitwise_and(red_mask, valid_mask, red_mask);

    if (thr.morph_open_k > 1) {
        const int k = (thr.morph_open_k % 2 == 0) ? thr.morph_open_k + 1 : thr.morph_open_k;
        cv::Mat se = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
        cv::morphologyEx(red_mask, red_mask, cv::MORPH_OPEN, se);
    }
    if (thr.morph_close_k > 1) {
        const int k = (thr.morph_close_k % 2 == 0) ? thr.morph_close_k + 1 : thr.morph_close_k;
        cv::Mat se = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
        cv::morphologyEx(red_mask, red_mask, cv::MORPH_CLOSE, se);
    }

    out.red_mask_full = red_mask;
    out.upper = measure_zone(out.red_mask_full, valid_mask, out.upper_zone, thr);
    out.lower = measure_zone(out.red_mask_full, valid_mask, out.lower_zone, thr);
    out.band = measure_zone(out.red_mask_full, valid_mask, out.full_search_rect, thr);
    out.band.pass = (out.band.red_pixels >= thr.band_min_pixels) && (out.band.red_ratio >= thr.band_min_ratio);
    return true;
}

bool evaluate_stacked_trigger(const StackedRedDebug& dbg,
                              const DynamicStackedConfig& dyn,
                              const RedThresholdConfig& thr,
                              TriggerState& state,
                              TriggerResult& out,
                              std::string& err) {
    out = {};
    err.clear();

    out.upper_pass = dbg.upper.pass;
    out.lower_pass = dbg.lower.pass;
    out.band_pass = dbg.band.pass;
    out.x_upper = dbg.upper.x_center;
    out.x_lower = dbg.lower.x_center;

    const bool have_x = (dbg.upper.x_center >= 0.f) && (dbg.lower.x_center >= 0.f);
    if (have_x) {
        out.x_consistent = std::abs(dbg.upper.x_center - dbg.lower.x_center) <= thr.center_x_max_diff;
        if (out.x_consistent) {
            out.x_center_raw = 0.5f * (dbg.upper.x_center + dbg.lower.x_center);
        }
    }

    const bool frame_good = out.upper_pass && out.lower_pass && out.band_pass && out.x_consistent;
    if (frame_good) {
        state.consecutive_good += 1;
        state.miss_count = 0;
        if (!state.had_valid_x || state.smoothed_x < 0.f) {
            state.smoothed_x = out.x_center_raw;
        } else {
            const float a = static_cast<float>(std::clamp(dyn.x_smoothing_alpha, 0.0, 1.0));
            state.smoothed_x = a * state.smoothed_x + (1.f - a) * out.x_center_raw;
        }
        state.had_valid_x = true;
    } else {
        state.consecutive_good = 0;
        if (state.had_valid_x && state.miss_count < dyn.miss_tolerance_frames) {
            state.miss_count += 1;
        } else {
            state.had_valid_x = false;
            state.smoothed_x = -1.f;
            state.miss_count = 0;
        }
    }

    out.consecutive_good = state.consecutive_good;
    out.x_center = state.had_valid_x ? state.smoothed_x : -1.f;
    out.trigger_ready = frame_good && (state.consecutive_good >= thr.trigger_consecutive_frames);
    return true;
}

} // namespace vision_app
