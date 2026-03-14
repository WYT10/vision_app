#include "app_config.hpp"
#include <iostream>

bool AppConfig::load(const std::string &filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
        return false;

    cv::FileNode cam = fs["camera"];
    cam["id"] >> camera_id;
    cam["width"] >> width;
    cam["height"] >> height;
    cam["fps"] >> fps;

    cv::FileNode calib = fs["calibration"];
    calib["homography"] >> homography;
    calib["warp_width"] >> warp_width;
    calib["warp_height"] >> warp_height;

    std::vector<float> rl, yo;
    calib["red_line_roi"] >> rl;
    if (rl.size() == 4)
        red_line_roi = cv::Rect2f(rl[0], rl[1], rl[2], rl[3]);

    calib["yolo_roi"] >> yo;
    if (yo.size() == 4)
        yolo_roi = cv::Rect2f(yo[0], yo[1], yo[2], yo[3]);

    fs.release();
    return true;
}

bool AppConfig::save(const std::string &filename) const
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened())
        return false;

    fs << "camera" << "{";
    fs << "id" << camera_id;
    fs << "width" << width;
    fs << "height" << height;
    fs << "fps" << fps;
    fs << "}";

    fs << "calibration" << "{";
    fs << "homography" << homography;
    fs << "warp_width" << warp_width;
    fs << "warp_height" << warp_height;

    std::vector<float> rl = {red_line_roi.x, red_line_roi.y, red_line_roi.width, red_line_roi.height};
    fs << "red_line_roi" << rl;

    std::vector<float> yo = {yolo_roi.x, yolo_roi.y, yolo_roi.width, yolo_roi.height};
    fs << "yolo_roi" << yo;
    fs << "}";

    fs.release();
    return true;
}