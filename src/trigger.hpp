
#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "calibrate.hpp"
#include "model.hpp"

namespace vision_app {

struct RedThresholdConfig {
    int h1_low = 0;
    int h1_high = 10;
    int h2_low = 170;
    int h2_high = 180;
    int s_min = 80;
    int v_min = 60;
};

struct BandRatio {
    double y = 0.24;
    double h = 0.08;
};

struct FixedRectTriggerConfig {
    double red_ratio_threshold = 0.18;
};

struct DynamicImageRoiConfig {
    double bottom_offset = 0.05; // normalized distance from upper band top to image ROI bottom
    double width = 0.32;
    double height = 0.32;
};

struct DynamicRedStackedConfig {
    BandRatio upper_band{0.24, 0.08};
    BandRatio lower_band{0.44, 0.08};
    double min_red_width_ratio = 0.18;
    double min_red_fill_ratio = 0.12;
    double x_smoothing_alpha = 0.35;
    DynamicImageRoiConfig image_roi;
};

struct TriggerDebugInfo {
    bool triggered = false;
    double red_ratio = 0.0;

    cv::Rect upper_band_px;
    cv::Rect lower_band_px;
    cv::Rect derived_image_roi_px;

    double upper_fill_ratio = 0.0;
    double lower_fill_ratio = 0.0;
    double upper_width_ratio = 0.0;
    double lower_width_ratio = 0.0;

    int upper_x0 = -1;
    int upper_x1 = -1;
    int lower_x0 = -1;
    int lower_x1 = -1;
    int center_x_px = -1;

    std::string summary;
};

bool extract_runtime_rois_fixed(const cv::Mat& warped,
                                const cv::Mat& valid_mask,
                                const RoiConfig& rois,
                                const RedThresholdConfig& red_cfg,
                                const FixedRectTriggerConfig& fixed_cfg,
                                RoiRuntimeData& out,
                                TriggerDebugInfo& dbg,
                                std::string& err);

bool extract_runtime_rois_dynamic(const cv::Mat& warped,
                                  const cv::Mat& valid_mask,
                                  const DynamicRedStackedConfig& dyn_cfg,
                                  const RedThresholdConfig& red_cfg,
                                  RoiRuntimeData& out,
                                  TriggerDebugInfo& dbg,
                                  std::string& err);

void draw_trigger_overlay_fixed(cv::Mat& img,
                                const RoiConfig& rois,
                                const TriggerDebugInfo& dbg,
                                bool highlight_red,
                                bool highlight_image);

void draw_trigger_overlay_dynamic(cv::Mat& img,
                                  const TriggerDebugInfo& dbg,
                                  bool highlight_upper,
                                  bool highlight_lower,
                                  bool highlight_image);

void adjust_band(BandRatio& band, int key, double move_step, double size_step);
void adjust_dynamic_image_roi(DynamicImageRoiConfig& cfg, int key, double move_step, double size_step);

} // namespace vision_app
