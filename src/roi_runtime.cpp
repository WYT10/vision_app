#include "roi_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace vision_app {
namespace {

static cv::Rect clamp_rect(const cv::Rect& r, const cv::Size& sz) {
    const int x0 = std::clamp(r.x, 0, std::max(0, sz.width));
    const int y0 = std::clamp(r.y, 0, std::max(0, sz.height));
    const int x1 = std::clamp(r.x + r.width, 0, std::max(0, sz.width));
    const int y1 = std::clamp(r.y + r.height, 0, std::max(0, sz.height));
    return cv::Rect(x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0));
}

static cv::Rect ratio_to_rect(double x, double y, double w, double h, const cv::Size& sz) {
    int rx = static_cast<int>(std::round(x * sz.width));
    int ry = static_cast<int>(std::round(y * sz.height));
    int rw = static_cast<int>(std::round(w * sz.width));
    int rh = static_cast<int>(std::round(h * sz.height));
    return clamp_rect(cv::Rect(rx, ry, rw, rh), sz);
}

} // namespace

bool synthesize_fixed_rois(const cv::Size& warped_size,
                           const FixedRoiConfig& fixed_cfg,
                           RuntimeRoiResult& out,
                           std::string& err) {
    out = {};
    err.clear();
    if (warped_size.width <= 0 || warped_size.height <= 0) {
        err = "invalid warped size";
        return false;
    }
    out.red_rect = ratio_to_rect(fixed_cfg.red_x, fixed_cfg.red_y, fixed_cfg.red_w, fixed_cfg.red_h, warped_size);
    out.image_rect = ratio_to_rect(fixed_cfg.img_x, fixed_cfg.img_y, fixed_cfg.img_w, fixed_cfg.img_h, warped_size);
    out.valid = out.red_rect.width > 0 && out.red_rect.height > 0 && out.image_rect.width > 0 && out.image_rect.height > 0;
    out.trigger_ready = true;
    out.summary = "fixed roi mode";
    return out.valid;
}

bool synthesize_dynamic_roi_above_upper(const cv::Size& warped_size,
                                        const DynamicStackedConfig& dyn,
                                        const TriggerResult& trig,
                                        RuntimeRoiResult& out,
                                        std::string& err) {
    out = {};
    err.clear();
    if (warped_size.width <= 0 || warped_size.height <= 0) {
        err = "invalid warped size";
        return false;
    }
    if (!trig.trigger_ready || trig.x_center < 0.f) {
        out.summary = "trigger not ready";
        return true;
    }

    const int search_x1 = (dyn.search_x1 < 0) ? warped_size.width : std::clamp(dyn.search_x1, 0, warped_size.width);
    const int search_x0 = std::clamp(dyn.search_x0, 0, warped_size.width);
    const int upper_y0 = std::clamp(dyn.upper_y0, 0, warped_size.height);
    const int lower_y1 = std::clamp(dyn.lower_y1, 0, warped_size.height);

    out.red_rect = clamp_rect(cv::Rect(search_x0, upper_y0, std::max(0, search_x1 - search_x0), std::max(0, lower_y1 - upper_y0)), warped_size);

    const int roi_w = std::max(1, dyn.roi_width);
    const int roi_h = std::max(1, dyn.roi_height);
    const int roi_bottom = upper_y0 - dyn.roi_gap_above_upper_zone;
    const int roi_top = roi_bottom - roi_h;
    const int roi_left = static_cast<int>(std::round(trig.x_center - 0.5f * roi_w));

    out.image_rect = clamp_rect(cv::Rect(roi_left, roi_top, roi_w, roi_h), warped_size);
    out.valid = out.image_rect.width > 0 && out.image_rect.height > 0;
    out.trigger_ready = trig.trigger_ready;
    out.x_center = trig.x_center;

    std::ostringstream oss;
    oss << "dynamic x=" << trig.x_center << " roi_above_upper";
    out.summary = oss.str();
    return true;
}

} // namespace vision_app
