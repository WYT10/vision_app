#ifndef APP_CONFIG_HPP
#define APP_CONFIG_HPP

#include <opencv2/opencv.hpp>
#include <string>

struct AppConfig
{
    int camera_id = 0;
    int width = 640;
    int height = 480;
    int fps = 60;

    cv::Mat homography;
    int warp_width = 600;  // Adaptively calculated
    int warp_height = 600; // Adaptively calculated

    cv::Rect2f red_line_roi;
    cv::Rect2f yolo_roi;

    bool load(const std::string &filename);
    bool save(const std::string &filename) const;
};

#endif