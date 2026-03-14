#include "warp.hpp"

#include <algorithm>

#include <opencv2/imgproc.hpp>

#include "utils.hpp"

namespace app {

cv::Rect denormalizeRoi(const RoiNorm& roi, int width, int height) {
    int x = std::clamp(static_cast<int>(std::round(roi.x * width)), 0, width - 1);
    int y = std::clamp(static_cast<int>(std::round(roi.y * height)), 0, height - 1);
    int w = std::clamp(static_cast<int>(std::round(roi.w * width)), 1, width - x);
    int h = std::clamp(static_cast<int>(std::round(roi.h * height)), 1, height - y);
    return cv::Rect(x, y, w, h);
}

RoiNorm normalizeRect(const cv::Rect& rect, int width, int height) {
    RoiNorm roi;
    roi.x = clamp01(static_cast<double>(rect.x) / width);
    roi.y = clamp01(static_cast<double>(rect.y) / height);
    roi.w = clamp01(static_cast<double>(rect.width) / width);
    roi.h = clamp01(static_cast<double>(rect.height) / height);
    return roi;
}

std::pair<cv::Mat, cv::Size> buildHomographyAndSize(const std::vector<cv::Point2f>& src_corners) {
    const double top = cv::norm(src_corners[0] - src_corners[1]);
    const double right = cv::norm(src_corners[1] - src_corners[2]);
    const double bottom = cv::norm(src_corners[2] - src_corners[3]);
    const double left = cv::norm(src_corners[3] - src_corners[0]);

    const int out_w = std::max(32, static_cast<int>(std::round(std::max(top, bottom))));
    const int out_h = std::max(32, static_cast<int>(std::round(std::max(left, right))));

    std::vector<cv::Point2f> dst = {
        {0.0f, 0.0f},
        {static_cast<float>(out_w - 1), 0.0f},
        {static_cast<float>(out_w - 1), static_cast<float>(out_h - 1)},
        {0.0f, static_cast<float>(out_h - 1)}
    };

    return {cv::getPerspectiveTransform(src_corners, dst), cv::Size(out_w, out_h)};
}

cv::Mat warpFrame(const cv::Mat& frame, const cv::Mat& H, const cv::Size& out_size) {
    cv::Mat warped;
    cv::warpPerspective(frame, warped, H, out_size, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return warped;
}

cv::Mat calibrationHomographyToMat(const CalibrationData& calibration) {
    cv::Mat H(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            H.at<double>(r, c) = calibration.H[r * 3 + c];
        }
    }
    return H;
}

} // namespace app
