#ifndef APP_CAMERA_HPP
#define APP_CAMERA_HPP

#include <opencv2/opencv.hpp>

struct CameraConfig
{
    int device_id;
    int width;
    int height;
    int fps;
};

class AppCamera
{
public:
    AppCamera();
    ~AppCamera();
    bool open(const CameraConfig &config);
    bool read(cv::Mat &frame);
    void close();
    bool isOpened() const;

private:
    cv::VideoCapture cap_;
};

#endif