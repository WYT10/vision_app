#pragma once
#include "config.h"
#include <opencv2/aruco.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

namespace app {

struct TagDetection {
    bool found = false;
    int id = -1;
    std::vector<cv::Point2f> corners;
};

class HomographyEngine {
public:
    explicit HomographyEngine(const RemapConfig& cfg);

    bool detectTag(const cv::Mat& frame_bgr, TagDetection& out, std::string* err = nullptr);
    bool calculateHomography(const TagDetection& det, cv::Mat& H, std::string* err = nullptr);
    bool computeWarpedSize(const cv::Mat& src_frame, const cv::Mat& H, cv::Size& out_size, std::string* err = nullptr) const;
    bool warpImage(const cv::Mat& src_frame, const cv::Mat& H, const cv::Size& out_size, cv::Mat& warped, std::string* err = nullptr) const;

    static bool validateWarpSize(const cv::Size& size);
    static cv::Mat makePreview255(const cv::Mat& warped);
    static cv::Rect roiFromRatio(const RoiRatio& ratio, const cv::Size& size);
    static RoiRatio ratioFromRect(const cv::Rect& rect, const cv::Size& size);

private:
    cv::aruco::PredefinedDictionaryType mapTagFamily(const std::string& tag_family) const;

private:
    RemapConfig cfg_;
    cv::Ptr<cv::aruco::Dictionary> dict_;
    cv::aruco::DetectorParameters params_;
    cv::aruco::ArucoDetector detector_;
};

} // namespace app
