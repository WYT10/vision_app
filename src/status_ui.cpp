
#include "status_ui.hpp"

#include <opencv2/imgproc.hpp>

namespace vision_app {

cv::Mat build_status_panel(const std::string& title,
                           const std::vector<std::string>& lines,
                           int width,
                           int line_height,
                           int margin) {
    const int body_lines = static_cast<int>(lines.size());
    const int height = std::max(140, margin * 2 + 40 + body_lines * line_height);
    cv::Mat panel(height, std::max(320, width), CV_8UC3, cv::Scalar(28, 28, 28));

    cv::rectangle(panel, cv::Rect(0, 0, panel.cols, 44), cv::Scalar(40, 40, 40), -1);
    cv::putText(panel, title, {margin, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(245,245,245), 2, cv::LINE_AA);

    int y = 44 + margin;
    for (const auto& line : lines) {
        cv::putText(panel, line, {margin, y}, cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(220,220,220), 1, cv::LINE_AA);
        y += line_height;
    }
    return panel;
}

} // namespace vision_app
