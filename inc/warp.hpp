#pragma once

#include <utility>

#include "types.hpp"

namespace app {

cv::Rect denormalizeRoi(const RoiNorm& roi, int width, int height);
RoiNorm normalizeRect(const cv::Rect& rect, int width, int height);
std::pair<cv::Mat, cv::Size> buildHomographyAndSize(const std::vector<cv::Point2f>& src_corners);
cv::Mat warpFrame(const cv::Mat& frame, const cv::Mat& H, const cv::Size& out_size);
cv::Mat calibrationHomographyToMat(const CalibrationData& calibration);

} // namespace app
