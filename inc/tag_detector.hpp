#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <vector>
#include <string>

namespace app
{

    struct Detection
    {
        int id;
        std::vector<cv::Point2f> corners;
    };

    struct TagSpec
    {
        bool auto_mode = true;
        std::string family = "36h11";
        int id = -1;
    };

    cv::aruco::Dictionary dictionaryFromFamily(const std::string &family);

    std::vector<Detection> detectTags(const cv::Mat &frame);

    const Detection *selectDetection(
        const std::vector<Detection> &detections,
        const TagSpec &spec);

}