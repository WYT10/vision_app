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
    int warp_width = 600;
    int warp_height = 600;

    cv::Rect2f red_line_roi;
    cv::Rect2f yolo_roi;

    int h_min = 0, s_min = 0, v_min = 0;
    int h_max = 179, s_max = 255, v_max = 255;

    bool load(const std::string &path);
    bool save(const std::string &path) const;
};
#endif