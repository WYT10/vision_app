#include "homography.h"
#include <algorithm>
#include <cmath>

namespace app {

cv::aruco::PredefinedDictionaryType HomographyEngine::mapTagFamily(const std::string& tag_family) const {
    if (tag_family == "16h5") return cv::aruco::DICT_APRILTAG_16h5;
    if (tag_family == "25h9") return cv::aruco::DICT_APRILTAG_25h9;
    return cv::aruco::DICT_APRILTAG_36h11;
}

HomographyEngine::HomographyEngine(const RemapConfig& cfg)
    : cfg_(cfg),
      dict_(cv::aruco::getPredefinedDictionary(mapTagFamily(cfg.tag_family))),
      params_(),
      detector_(dict_, params_) {}

bool HomographyEngine::detectTag(const cv::Mat& frame_bgr, TagDetection& out, std::string* err) {
    out = {};
    if (frame_bgr.empty()) {
        if (err) *err = "empty input frame";
        return false;
    }
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    detector_.detectMarkers(frame_bgr, corners, ids);
    if (ids.empty() || corners.empty()) return true;
    out.found = true;
    out.id = ids[0];
    out.corners = corners[0];
    return true;
}

bool HomographyEngine::calculateHomography(const TagDetection& det, cv::Mat& H, std::string* err) {
    if (!det.found || det.corners.size() != 4) {
        if (err) *err = "no valid tag corners";
        return false;
    }
    const std::vector<cv::Point2f> dst = {
        {0.0f, 0.0f}, {200.0f, 0.0f}, {200.0f, 200.0f}, {0.0f, 200.0f}
    };
    H = cv::findHomography(det.corners, dst);
    if (H.empty() || H.rows != 3 || H.cols != 3) {
        if (err) *err = "findHomography failed";
        return false;
    }
    for (int r = 0; r < H.rows; ++r) {
        for (int c = 0; c < H.cols; ++c) {
            if (!std::isfinite(H.at<double>(r, c))) {
                if (err) *err = "non-finite homography value";
                return false;
            }
        }
    }
    return true;
}

bool HomographyEngine::computeWarpedSize(const cv::Mat& src_frame, const cv::Mat& H, cv::Size& out_size, std::string* err) const {
    if (src_frame.empty() || H.empty()) {
        if (err) *err = "missing frame or H";
        return false;
    }
    std::vector<cv::Point2f> src = {
        {0.0f, 0.0f},
        {static_cast<float>(src_frame.cols - 1), 0.0f},
        {static_cast<float>(src_frame.cols - 1), static_cast<float>(src_frame.rows - 1)},
        {0.0f, static_cast<float>(src_frame.rows - 1)}
    };
    std::vector<cv::Point2f> dst;
    cv::perspectiveTransform(src, dst, H);
    float min_x = dst[0].x, max_x = dst[0].x, min_y = dst[0].y, max_y = dst[0].y;
    for (const auto& p : dst) {
        min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
    }
    const int w = static_cast<int>(std::ceil(max_x - min_x));
    const int h = static_cast<int>(std::ceil(max_y - min_y));
    out_size = cv::Size(w, h);
    if (!validateWarpSize(out_size)) {
        if (err) *err = "unsafe warp output size";
        return false;
    }
    return true;
}

bool HomographyEngine::warpImage(const cv::Mat& src_frame, const cv::Mat& H, const cv::Size& out_size, cv::Mat& warped, std::string* err) const {
    if (!validateWarpSize(out_size)) {
        if (err) *err = "invalid warp size";
        return false;
    }
    cv::warpPerspective(src_frame, warped, H, out_size, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return !warped.empty();
}

bool HomographyEngine::validateWarpSize(const cv::Size& size) {
    return size.width > 0 && size.height > 0 && size.width <= 2048 && size.height <= 2048;
}

cv::Mat HomographyEngine::makePreview255(const cv::Mat& warped) {
    if (warped.empty()) return {};
    const int side = std::max(warped.cols, warped.rows);
    cv::Mat square(side, side, warped.type(), cv::Scalar::all(0));
    const int x = (side - warped.cols) / 2;
    const int y = (side - warped.rows) / 2;
    warped.copyTo(square(cv::Rect(x, y, warped.cols, warped.rows)));
    cv::Mat preview;
    cv::resize(square, preview, cv::Size(255, 255), 0, 0, cv::INTER_AREA);
    return preview;
}

cv::Rect HomographyEngine::roiFromRatio(const RoiRatio& ratio, const cv::Size& size) {
    int x = static_cast<int>(std::round(ratio.x * size.width));
    int y = static_cast<int>(std::round(ratio.y * size.height));
    int w = static_cast<int>(std::round(ratio.w * size.width));
    int h = static_cast<int>(std::round(ratio.h * size.height));
    x = std::clamp(x, 0, std::max(0, size.width - 1));
    y = std::clamp(y, 0, std::max(0, size.height - 1));
    w = std::clamp(w, 1, std::max(1, size.width - x));
    h = std::clamp(h, 1, std::max(1, size.height - y));
    return {x, y, w, h};
}

RoiRatio HomographyEngine::ratioFromRect(const cv::Rect& rect, const cv::Size& size) {
    RoiRatio r;
    if (size.width <= 0 || size.height <= 0) return r;
    r.x = static_cast<double>(rect.x) / size.width;
    r.y = static_cast<double>(rect.y) / size.height;
    r.w = static_cast<double>(rect.width) / size.width;
    r.h = static_cast<double>(rect.height) / size.height;
    return r;
}

} // namespace app
