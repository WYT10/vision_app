#pragma once
#include <opencv2/core.hpp>
#include <string>
#include "params.hpp"

namespace vision_app {

struct ZoneStats {
    int valid_pixels = 0;
    int red_pixels = 0;
    double red_ratio = 0.0;
    bool pass = false;
    cv::Rect best_blob_bbox;
    float x_center = -1.f;
};

struct StackedRedDebug {
    cv::Rect upper_zone;
    cv::Rect lower_zone;
    cv::Rect full_search_rect;
    cv::Mat red_mask_full; // full warped coordinates
    ZoneStats upper;
    ZoneStats lower;
    ZoneStats band;
};

struct TriggerState {
    int consecutive_good = 0;
    int miss_count = 0;
    float smoothed_x = -1.f;
    bool had_valid_x = false;
};

struct TriggerResult {
    bool upper_pass = false;
    bool lower_pass = false;
    bool band_pass = false;
    bool x_consistent = false;
    bool trigger_ready = false;
    int consecutive_good = 0;

    float x_upper = -1.f;
    float x_lower = -1.f;
    float x_center_raw = -1.f;
    float x_center = -1.f;
};

bool compute_red_mask_stacked(const cv::Mat& warped,
                              const cv::Mat& valid_mask,
                              const DynamicStackedConfig& dyn,
                              const RedThresholdConfig& thr,
                              StackedRedDebug& out,
                              std::string& err);

bool evaluate_stacked_trigger(const StackedRedDebug& dbg,
                              const DynamicStackedConfig& dyn,
                              const RedThresholdConfig& thr,
                              TriggerState& state,
                              TriggerResult& out,
                              std::string& err);

} // namespace vision_app
