#include "app_camera.hpp"
#include <iostream>

AppCamera::AppCamera() {}
AppCamera::~AppCamera() { close(); }

bool AppCamera::open(const CameraConfig &config)
{
    cap_.open(config.device_id, cv::CAP_V4L2);
    if (!cap_.isOpened())
        return false;

    cap_.set(cv::CAP_PROP_FRAME_WIDTH, config.width);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, config.height);
    cap_.set(cv::CAP_PROP_FPS, config.fps);
    return true;
}

bool AppCamera::read(cv::Mat &frame)
{
    if (!cap_.isOpened())
        return false;
    return cap_.read(frame);
}

void AppCamera::close()
{
    if (cap_.isOpened())
        cap_.release();
}

bool AppCamera::isOpened() const
{
    return cap_.isOpened();
}