#include "tag_detector.hpp"

namespace app
{

    cv::aruco::Dictionary dictionaryFromFamily(const std::string &f)
    {
        if (f.find("36h11") != std::string::npos)
            return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);

        if (f.find("25h9") != std::string::npos)
            return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_25h9);

        if (f.find("16h5") != std::string::npos)
            return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_16h5);

        if (f.find("36h10") != std::string::npos)
            return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h10);

        return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);
    }

    std::vector<Detection> detectTags(const cv::Mat &frame)
    {
        std::vector<Detection> detections;

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;

        auto dictionary =
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);

        cv::aruco::DetectorParameters params;

        cv::aruco::ArucoDetector detector(dictionary, params);

        detector.detectMarkers(frame, corners, ids);

        for (size_t i = 0; i < ids.size(); ++i)
        {
            Detection d;
            d.id = ids[i];
            d.corners = corners[i];
            detections.push_back(d);
        }

        return detections;
    }

    const Detection *selectDetection(
        const std::vector<Detection> &detections,
        const TagSpec &spec)
    {
        if (detections.empty())
            return nullptr;

        // AUTO MODE -> return first detected tag
        if (spec.auto_mode)
            return &detections[0];

        // ID specific
        if (spec.id >= 0)
        {
            for (const auto &d : detections)
            {
                if (d.id == spec.id)
                    return &d;
            }
        }

        // fallback
        return &detections[0];
    }

}