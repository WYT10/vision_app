
#include "trigger.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

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

cv::Mat build_red_mask(const cv::Mat& bgr, const cv::Mat& valid_mask, const RedThresholdConfig& red_cfg) {
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat m1, m2, red_mask;
    cv::inRange(hsv, cv::Scalar(red_cfg.h1_low, red_cfg.s_min, red_cfg.v_min), cv::Scalar(red_cfg.h1_high, 255, 255), m1);
    cv::inRange(hsv, cv::Scalar(red_cfg.h2_low, red_cfg.s_min, red_cfg.v_min), cv::Scalar(red_cfg.h2_high, 255, 255), m2);
    cv::bitwise_or(m1, m2, red_mask);
    if (!valid_mask.empty()) cv::bitwise_and(red_mask, valid_mask, red_mask);
    return red_mask;
}

cv::Rect clamp_rect(const cv::Rect& r, const cv::Size& sz) {
    const int x0 = std::clamp(r.x, 0, std::max(0, sz.width - 1));
    const int y0 = std::clamp(r.y, 0, std::max(0, sz.height - 1));
    const int x1 = std::clamp(r.x + r.width, x0 + 1, sz.width);
    const int y1 = std::clamp(r.y + r.height, y0 + 1, sz.height);
    return cv::Rect(x0, y0, x1 - x0, y1 - y0);
}

cv::Rect band_to_rect(const BandRatio& b, const cv::Size& sz) {
    int y = static_cast<int>(std::round(b.y * sz.height));
    int h = std::max(1, static_cast<int>(std::round(b.h * sz.height)));
    return clamp_rect(cv::Rect(0, y, sz.width, h), sz);
}

struct BandMetrics {
    double fill_ratio = 0.0;
    double width_ratio = 0.0;
    int x0 = -1;
    int x1 = -1;
    int center_x = -1;
};

BandMetrics analyze_band(const cv::Mat& red_mask, const cv::Mat& valid_mask) {
    BandMetrics m;
    const int valid_px = std::max(1, cv::countNonZero(valid_mask));
    const int red_px = cv::countNonZero(red_mask);
    m.fill_ratio = static_cast<double>(red_px) / static_cast<double>(valid_px);

    std::vector<int> active_cols;
    active_cols.reserve(red_mask.cols);
    for (int x = 0; x < red_mask.cols; ++x) {
        const int valid_col = std::max(1, cv::countNonZero(valid_mask.col(x)));
        const int red_col = cv::countNonZero(red_mask.col(x));
        const double col_fill = static_cast<double>(red_col) / static_cast<double>(valid_col);
        if (col_fill >= 0.10) active_cols.push_back(x);
    }
    if (!active_cols.empty()) {
        m.x0 = active_cols.front();
        m.x1 = active_cols.back();
        m.center_x = (m.x0 + m.x1) / 2;
        m.width_ratio = static_cast<double>(m.x1 - m.x0 + 1) / static_cast<double>(std::max(1, red_mask.cols));
    }
    return m;
}

} // namespace

bool extract_runtime_rois_fixed(const cv::Mat& warped,
                                const cv::Mat& valid_mask,
                                const RoiConfig& rois,
                                const RedThresholdConfig& red_cfg,
                                const FixedRectTriggerConfig& fixed_cfg,
                                RoiRuntimeData& out,
                                TriggerDebugInfo& dbg,
                                std::string& err) {
    out = {};
    dbg = {};
    if (warped.empty() || valid_mask.empty()) {
        err = "empty warped image or mask";
        return false;
    }

    const cv::Rect rr = roi_to_rect(rois.red_roi, warped.size());
    const cv::Rect ir = roi_to_rect(rois.image_roi, warped.size());
    dbg.derived_image_roi_px = ir;

    out.red_bgr = warped(rr).clone();
    out.red_mask = valid_mask(rr).clone();
    out.image_bgr = warped(ir).clone();
    out.image_mask = valid_mask(ir).clone();
    out.red_valid_pixels = cv::countNonZero(out.red_mask);
    out.image_valid_pixels = cv::countNonZero(out.image_mask);

    cv::Mat red_mask = build_red_mask(out.red_bgr, out.red_mask, red_cfg);
    out.red_ratio = static_cast<double>(cv::countNonZero(red_mask)) / static_cast<double>(std::max(1, out.red_valid_pixels));
    dbg.red_ratio = out.red_ratio;
    dbg.triggered = out.red_ratio >= fixed_cfg.red_ratio_threshold;

    out.image_bgr = apply_mask_fill(out.image_bgr, out.image_mask, cv::Scalar(255,255,255));

    std::ostringstream oss;
    oss << "fixed red_ratio=" << out.red_ratio << " thr=" << fixed_cfg.red_ratio_threshold
        << " triggered=" << (dbg.triggered ? "1" : "0");
    dbg.summary = oss.str();
    err.clear();
    return true;
}

bool extract_runtime_rois_dynamic(const cv::Mat& warped,
                                  const cv::Mat& valid_mask,
                                  const DynamicRedStackedConfig& dyn_cfg,
                                  const RedThresholdConfig& red_cfg,
                                  RoiRuntimeData& out,
                                  TriggerDebugInfo& dbg,
                                  std::string& err) {
    out = {};
    dbg = {};
    if (warped.empty() || valid_mask.empty()) {
        err = "empty warped image or mask";
        return false;
    }

    dbg.upper_band_px = band_to_rect(dyn_cfg.upper_band, warped.size());
    dbg.lower_band_px = band_to_rect(dyn_cfg.lower_band, warped.size());

    const cv::Mat upper_bgr = warped(dbg.upper_band_px).clone();
    const cv::Mat upper_valid = valid_mask(dbg.upper_band_px).clone();
    const cv::Mat lower_bgr = warped(dbg.lower_band_px).clone();
    const cv::Mat lower_valid = valid_mask(dbg.lower_band_px).clone();

    const cv::Mat upper_red = build_red_mask(upper_bgr, upper_valid, red_cfg);
    const cv::Mat lower_red = build_red_mask(lower_bgr, lower_valid, red_cfg);

    const BandMetrics upper = analyze_band(upper_red, upper_valid);
    const BandMetrics lower = analyze_band(lower_red, lower_valid);
    dbg.upper_fill_ratio = upper.fill_ratio;
    dbg.lower_fill_ratio = lower.fill_ratio;
    dbg.upper_width_ratio = upper.width_ratio;
    dbg.lower_width_ratio = lower.width_ratio;
    dbg.upper_x0 = upper.x0;
    dbg.upper_x1 = upper.x1;
    dbg.lower_x0 = lower.x0;
    dbg.lower_x1 = lower.x1;

    const bool upper_ok = upper.width_ratio >= dyn_cfg.min_red_width_ratio && upper.fill_ratio >= dyn_cfg.min_red_fill_ratio;
    const bool lower_ok = lower.width_ratio >= dyn_cfg.min_red_width_ratio && lower.fill_ratio >= dyn_cfg.min_red_fill_ratio;
    dbg.triggered = upper_ok && lower_ok;

    if (upper.center_x >= 0 && lower.center_x >= 0) dbg.center_x_px = (upper.center_x + lower.center_x) / 2;
    else dbg.center_x_px = std::max(upper.center_x, lower.center_x);

    // First-pass anchoring rule: place image ROI above the upper red band.
    const int roi_w = std::max(1, static_cast<int>(std::round(dyn_cfg.image_roi.width * warped.cols)));
    const int roi_h = std::max(1, static_cast<int>(std::round(dyn_cfg.image_roi.height * warped.rows)));
    const int cx = (dbg.center_x_px >= 0) ? dbg.center_x_px : warped.cols / 2;
    const int bottom = dbg.upper_band_px.y - static_cast<int>(std::round(dyn_cfg.image_roi.bottom_offset * warped.rows));
    const int x0 = cx - roi_w / 2;
    const int y0 = bottom - roi_h;
    dbg.derived_image_roi_px = clamp_rect(cv::Rect(x0, y0, roi_w, roi_h), warped.size());

    out.red_bgr = warped(dbg.upper_band_px | dbg.lower_band_px).clone();
    out.red_mask = valid_mask(dbg.upper_band_px | dbg.lower_band_px).clone();
    out.image_bgr = warped(dbg.derived_image_roi_px).clone();
    out.image_mask = valid_mask(dbg.derived_image_roi_px).clone();
    out.red_valid_pixels = cv::countNonZero(out.red_mask);
    out.image_valid_pixels = cv::countNonZero(out.image_mask);
    const cv::Mat all_red = build_red_mask(out.red_bgr, out.red_mask, red_cfg);
    out.red_ratio = static_cast<double>(cv::countNonZero(all_red)) / static_cast<double>(std::max(1, out.red_valid_pixels));
    dbg.red_ratio = out.red_ratio;
    out.image_bgr = apply_mask_fill(out.image_bgr, out.image_mask, cv::Scalar(255,255,255));

    std::ostringstream oss;
    oss << "dynamic trig=" << (dbg.triggered ? "1" : "0")
        << " upper(fill=" << dbg.upper_fill_ratio << ",w=" << dbg.upper_width_ratio << ")"
        << " lower(fill=" << dbg.lower_fill_ratio << ",w=" << dbg.lower_width_ratio << ")"
        << " cx=" << dbg.center_x_px;
    dbg.summary = oss.str();
    err.clear();
    return true;
}

void draw_trigger_overlay_fixed(cv::Mat& img,
                                const RoiConfig& rois,
                                const TriggerDebugInfo& dbg,
                                bool highlight_red,
                                bool highlight_image) {
    const cv::Rect rr = roi_to_rect(rois.red_roi, img.size());
    const cv::Rect ir = roi_to_rect(rois.image_roi, img.size());
    cv::rectangle(img, rr, highlight_red ? cv::Scalar(0,0,255) : cv::Scalar(40,40,180), 2);
    cv::putText(img, "red_roi", rr.tl() + cv::Point(4, 18), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0,0,255), 2);
    cv::rectangle(img, ir, highlight_image ? cv::Scalar(255,0,0) : cv::Scalar(180,40,40), 2);
    cv::putText(img, "image_roi", ir.tl() + cv::Point(4, 18), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255,0,0), 2);
    if (dbg.triggered) cv::putText(img, "TRIGGER", {12, img.rows - 18}, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);
}

void draw_trigger_overlay_dynamic(cv::Mat& img,
                                  const TriggerDebugInfo& dbg,
                                  bool highlight_upper,
                                  bool highlight_lower,
                                  bool highlight_image) {
    cv::rectangle(img, dbg.upper_band_px, highlight_upper ? cv::Scalar(0,0,255) : cv::Scalar(0,0,180), 2);
    cv::putText(img, "upper_band", dbg.upper_band_px.tl() + cv::Point(4, 18), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0,0,255), 2);
    cv::rectangle(img, dbg.lower_band_px, highlight_lower ? cv::Scalar(0,140,255) : cv::Scalar(0,120,180), 2);
    cv::putText(img, "lower_band", dbg.lower_band_px.tl() + cv::Point(4, 18), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0,140,255), 2);
    cv::rectangle(img, dbg.derived_image_roi_px, highlight_image ? cv::Scalar(255,0,0) : cv::Scalar(180,40,40), 2);
    cv::putText(img, "image_roi", dbg.derived_image_roi_px.tl() + cv::Point(4, 18), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255,0,0), 2);
    if (dbg.center_x_px >= 0) {
        cv::line(img, cv::Point(dbg.center_x_px, 0), cv::Point(dbg.center_x_px, img.rows - 1), cv::Scalar(0,255,255), 1, cv::LINE_AA);
    }
    if (dbg.triggered) cv::putText(img, "TRIGGER", {12, img.rows - 18}, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);
}

void adjust_band(BandRatio& band, int key, double move_step, double size_step) {
    if (key == 'w') band.y -= move_step;
    if (key == 's') band.y += move_step;
    if (key == 'i') band.h -= size_step;
    if (key == 'k') band.h += size_step;
    band.y = std::clamp(band.y, 0.0, 0.98);
    band.h = std::clamp(band.h, 0.01, 0.98 - band.y);
}

void adjust_dynamic_image_roi(DynamicImageRoiConfig& cfg, int key, double move_step, double size_step) {
    if (key == 'w') cfg.bottom_offset += move_step;
    if (key == 's') cfg.bottom_offset -= move_step;
    if (key == 'j') cfg.width -= size_step;
    if (key == 'l') cfg.width += size_step;
    if (key == 'i') cfg.height -= size_step;
    if (key == 'k') cfg.height += size_step;
    cfg.bottom_offset = std::clamp(cfg.bottom_offset, 0.0, 0.95);
    cfg.width = std::clamp(cfg.width, 0.02, 1.0);
    cfg.height = std::clamp(cfg.height, 0.02, 1.0);
}

} // namespace vision_app
