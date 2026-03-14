#ifndef APP_CAMERA_HPP
#define APP_CAMERA_HPP
#include <opencv2/opencv.hpp>
#include "app_config.hpp"

class AppCamera
{
public:
    AppCamera();
    ~AppCamera();
    bool open(const AppConfig &config);
    bool read(cv::Mat &frame);

private:
    cv::VideoCapture cap_;
};
#endif