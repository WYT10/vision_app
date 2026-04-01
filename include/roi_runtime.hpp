#pragma once
#include <opencv2/core.hpp>
#include <string>
#include "params.hpp"
#include "red_trigger.hpp"

namespace vision_app {

struct RuntimeRoiResult {
    bool valid = false;
    bool trigger_ready = false;
    cv::Rect red_rect;   // fixed mode baseline or representative trigger area
    cv::Rect image_rect; // final ROI that would feed the model
    float x_center = -1.f;
    std::string summary;
};

bool synthesize_fixed_rois(const cv::Size& warped_size,
                           const FixedRoiConfig& fixed_cfg,
                           RuntimeRoiResult& out,
                           std::string& err);

bool synthesize_dynamic_roi_above_upper(const cv::Size& warped_size,
                                        const DynamicStackedConfig& dyn,
                                        const TriggerResult& trig,
                                        RuntimeRoiResult& out,
                                        std::string& err);

} // namespace vision_app
