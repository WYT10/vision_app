#include "homography.h"

#include <algorithm>
#include <cmath>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace
{
std::vector<cv::Point2f> make_plane_tag_corners(double tag_size)
{
    return {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(static_cast<float>(tag_size), 0.0f),
        cv::Point2f(static_cast<float>(tag_size), static_cast<float>(tag_size)),
        cv::Point2f(0.0f, static_cast<float>(tag_size))
    };
}
}

bool compute_homography_from_tag(
    const std::vector<cv::Point2f>& detected_corners,
    const cv::Size& input_size,
    double tag_size_units,
    double output_padding_units,
    cv::Mat& adjusted_homography,
    cv::Size& warped_size)
{
    if (detected_corners.size() != 4 || input_size.width <= 0 || input_size.height <= 0)
        return false;

    const auto tag_plane = make_plane_tag_corners(tag_size_units);
    cv::Mat H_plane = cv::findHomography(detected_corners, tag_plane);
    if (H_plane.empty())
        return false;

    const std::vector<cv::Point2f> frame_corners = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(static_cast<float>(input_size.width - 1), 0.0f),
        cv::Point2f(static_cast<float>(input_size.width - 1), static_cast<float>(input_size.height - 1)),
        cv::Point2f(0.0f, static_cast<float>(input_size.height - 1))
    };

    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(frame_corners, projected, H_plane);

    float min_x = projected[0].x;
    float min_y = projected[0].y;
    float max_x = projected[0].x;
    float max_y = projected[0].y;

    for (const auto& p : projected)
    {
        min_x = std::min(min_x, p.x);
        min_y = std::min(min_y, p.y);
        max_x = std::max(max_x, p.x);
        max_y = std::max(max_y, p.y);
    }

    min_x -= static_cast<float>(output_padding_units);
    min_y -= static_cast<float>(output_padding_units);
    max_x += static_cast<float>(output_padding_units);
    max_y += static_cast<float>(output_padding_units);

    const double out_w = std::ceil(max_x - min_x);
    const double out_h = std::ceil(max_y - min_y);
    if (out_w <= 1.0 || out_h <= 1.0)
        return false;

    cv::Mat T = (cv::Mat_<double>(3, 3) <<
        1.0, 0.0, -static_cast<double>(min_x),
        0.0, 1.0, -static_cast<double>(min_y),
        0.0, 0.0, 1.0);

    adjusted_homography = T * H_plane;
    adjusted_homography.convertTo(adjusted_homography, CV_64F);
    warped_size = cv::Size(static_cast<int>(out_w), static_cast<int>(out_h));
    return true;
}

bool warp_frame(const cv::Mat& input, cv::Mat& output, const cv::Mat& homography, const cv::Size& warped_size)
{
    if (input.empty() || homography.empty() || warped_size.width <= 0 || warped_size.height <= 0)
        return false;

    cv::warpPerspective(input, output, homography, warped_size, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return !output.empty();
}

RoiRatio rect_to_ratio(const cv::Rect& rect, const cv::Size& bounds)
{
    RoiRatio out;
    if (bounds.width <= 0 || bounds.height <= 0 || rect.width <= 0 || rect.height <= 0)
        return out;

    out.x = static_cast<double>(rect.x) / static_cast<double>(bounds.width);
    out.y = static_cast<double>(rect.y) / static_cast<double>(bounds.height);
    out.w = static_cast<double>(rect.width) / static_cast<double>(bounds.width);
    out.h = static_cast<double>(rect.height) / static_cast<double>(bounds.height);
    return out;
}

cv::Rect ratio_to_rect_clamped(const RoiRatio& ratio, const cv::Size& bounds)
{
    if (bounds.width <= 0 || bounds.height <= 0)
        return {};

    int x = static_cast<int>(std::round(ratio.x * bounds.width));
    int y = static_cast<int>(std::round(ratio.y * bounds.height));
    x = std::clamp(x, 0, bounds.width - 1);
    y = std::clamp(y, 0, bounds.height - 1);

    int w = static_cast<int>(std::round(ratio.w * bounds.width));
    int h = static_cast<int>(std::round(ratio.h * bounds.height));
    w = std::max(1, w);
    h = std::max(1, h);

    w = std::min(w, bounds.width - x);
    h = std::min(h, bounds.height - y);
    return cv::Rect(x, y, w, h);
}

bool is_valid_ratio_roi(const RoiRatio& ratio)
{
    return ratio.w > 0.0 && ratio.h > 0.0;
}
