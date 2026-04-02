
#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace vision_app {

cv::Mat build_status_panel(const std::string& title,
                           const std::vector<std::string>& lines,
                           int width = 720,
                           int line_height = 22,
                           int margin = 14);

} // namespace vision_app
