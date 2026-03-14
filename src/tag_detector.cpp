#include "tag_detector.hpp"

#include <algorithm>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "utils.hpp"

namespace app {

cv::Ptr<cv::aruco::Dictionary> dictionaryFromFamily(const std::string& family) {
    const std::string f = lower(family);
    if (f.find("36h11") != std::string::npos) return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);
    if (f.find("25h9") != std::string::npos) return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_25h9);
    if (f.find("16h5") != std::string::npos) return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_16h5);
    if (f.find("36h10") != std::string::npos) return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h10);
    throw std::runtime_error("Unsupported AprilTag family: " + family);
}

std::vector<Detection> detectTags(const cv::Mat& frame_bgr) {
    std::vector<Detection> detections;
    const std::vector<std::pair<std::string, cv::aruco::PREDEFINED_DICTIONARY_NAME>> families = {
        {"AprilTag 36h11", cv::aruco::DICT_APRILTAG_36h11},
        {"AprilTag 25h9", cv::aruco::DICT_APRILTAG_25h9},
        {"AprilTag 16h5", cv::aruco::DICT_APRILTAG_16h5},
        {"AprilTag 36h10", cv::aruco::DICT_APRILTAG_36h10},
    };

    for (const auto& [family_name, family_id] : families) {
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        auto dictionary = cv::aruco::getPredefinedDictionary(family_id);
        auto params = cv::aruco::DetectorParameters::create();
        cv::aruco::detectMarkers(frame_bgr, dictionary, corners, ids, params);

        for (size_t i = 0; i < ids.size(); ++i) {
            detections.push_back(Detection{family_name, ids[i], corners[i]});
        }
    }

    return detections;
}

std::optional<Detection> selectDetection(const std::vector<Detection>& detections, const TagSpec& spec) {
    if (detections.empty()) return std::nullopt;

    if (spec.mode == "auto") {
        return *std::max_element(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
            return cv::contourArea(a.corners) < cv::contourArea(b.corners);
        });
    }

    for (const auto& detection : detections) {
        if (spec.mode == "family" && lower(detection.family) == lower(spec.family)) return detection;
        if (spec.mode == "id" && detection.id == spec.id) return detection;
    }

    return std::nullopt;
}

} // namespace app
