#pragma once

#include "config.h"
#include <opencv2/core.hpp>
#include <vector>

/*
==============================================================================
homography.h
==============================================================================
Purpose
    Shared geometry helpers used by both calibration and deploy.

Important safety note
    The warped output size is computed from projected frame bounds. That value
    must be checked before allocating a warped image, otherwise a bad detection
    can request a huge output buffer and destabilize a Pi.
==============================================================================
*/

/*
------------------------------------------------------------------------------
compute_homography_from_tag
------------------------------------------------------------------------------
Input
    detected_corners      : 4 tag corners in raw image coordinates
    input_size            : raw frame size
    tag_size_units        : size of the square tag plane in arbitrary units
    output_padding_units  : extra border around the projected full frame

Process
    1. Solve image -> tag plane homography.
    2. Project full frame corners into the plane.
    3. Compute bounded output image size.
    4. Add translation so the warped image starts at (0, 0).

Output
    adjusted_homography   : image -> warped plane transform
    warped_size           : output image size for warpPerspective
------------------------------------------------------------------------------
*/
bool compute_homography_from_tag(
    const std::vector<cv::Point2f>& detected_corners,
    const cv::Size& input_size,
    double tag_size_units,
    double output_padding_units,
    cv::Mat& adjusted_homography,
    cv::Size& warped_size);

/* Warp a frame using a precomputed homography. */
bool warp_frame(
    const cv::Mat& input,
    cv::Mat& output,
    const cv::Mat& homography,
    const cv::Size& warped_size);

/* Convert between pixel ROIs and normalized ratios. */
RoiRatio rect_to_ratio(const cv::Rect& rect, const cv::Size& bounds);
cv::Rect ratio_to_rect_clamped(const RoiRatio& ratio, const cv::Size& bounds);
bool is_valid_ratio_roi(const RoiRatio& ratio);

/*
------------------------------------------------------------------------------
Warp size guard
------------------------------------------------------------------------------
Input
    size : proposed warped image size

Return
    true only if the warp size is small enough to be considered safe for this
    application. The limit is conservative on purpose for Raspberry Pi use.
------------------------------------------------------------------------------
*/
bool is_safe_warp_size(const cv::Size& size);
