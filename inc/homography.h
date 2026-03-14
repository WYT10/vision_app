#pragma once

#include "config.h"
#include <opencv2/core.hpp>
#include <vector>

bool compute_homography_from_tag(
    const std::vector<cv::Point2f>& detected_corners,
    const cv::Size& input_size,
    double tag_size_units,
    double output_padding_units,
    cv::Mat& adjusted_homography,
    cv::Size& warped_size);

bool warp_frame(const cv::Mat& input, cv::Mat& output, const cv::Mat& homography, const cv::Size& warped_size);

RoiRatio rect_to_ratio(const cv::Rect& rect, const cv::Size& bounds);
cv::Rect ratio_to_rect_clamped(const RoiRatio& ratio, const cv::Size& bounds);
bool is_valid_ratio_roi(const RoiRatio& ratio);
