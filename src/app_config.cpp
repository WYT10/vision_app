#include "app_config.hpp"
#include <iostream>

bool AppConfig::load(const std::string &path)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened())
        return false;

    fs["camera_id"] >> camera_id;
    fs["width"] >> width;
    fs["height"] >> height;
    fs["fps"] >> fps;
    fs["homography"] >> homography;
    fs["warp_width"] >> warp_width;
    fs["warp_height"] >> warp_height;

    std::vector<float> rl, yo;
    fs["red_line_roi"] >> rl;
    if (rl.size() == 4)
        red_line_roi = cv::Rect2f(rl[0], rl[1], rl[2], rl[3]);

    fs["yolo_roi"] >> yo;
    if (yo.size() == 4)
        yolo_roi = cv::Rect2f(yo[0], yo[1], yo[2], yo[3]);

    fs["h_min"] >> h_min;
    fs["s_min"] >> s_min;
    fs["v_min"] >> v_min;
    fs["h_max"] >> h_max;
    fs["s_max"] >> s_max;
    fs["v_max"] >> v_max;

    fs.release();
    return true;
}

bool AppConfig::save(const std::string &path) const
{
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened())
        return false;

    fs << "camera_id" << camera_id << "width" << width << "height" << height << "fps" << fps;
    fs << "homography" << homography << "warp_width" << warp_width << "warp_height" << warp_height;

    fs << "red_line_roi" << std::vector<float>{red_line_roi.x, red_line_roi.y, red_line_roi.width, red_line_roi.height};
    fs << "yolo_roi" << std::vector<float>{yolo_roi.x, yolo_roi.y, yolo_roi.width, yolo_roi.height};

    fs << "h_min" << h_min << "s_min" << s_min << "v_min" << v_min;
    fs << "h_max" << h_max << "s_max" << s_max << "v_max" << v_max;

    fs.release();
    return true;
}