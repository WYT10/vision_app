#pragma once

#include <optional>
#include <vector>

#include <opencv2/aruco.hpp>

#include "types.hpp"

namespace app {

cv::Ptr<cv::aruco::Dictionary> dictionaryFromFamily(const std::string& family);
std::vector<Detection> detectTags(const cv::Mat& frame_bgr);
std::optional<Detection> selectDetection(const std::vector<Detection>& detections, const TagSpec& spec);

} // namespace app
