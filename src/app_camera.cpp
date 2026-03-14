#include "app_camera.hpp"

AppCamera::AppCamera() {}
AppCamera::~AppCamera()
{
    if (cap_.isOpened())
        cap_.release();
}

bool AppCamera::open(const AppConfig &config)
{
    cap_.open(config.camera_id, cv::CAP_V4L2);
    if (!cap_.isOpened())
        return false;

    cap_.set(cv::CAP_PROP_FRAME_WIDTH, config.width);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, config.height);
    cap_.set(cv::CAP_PROP_FPS, config.fps);

    // Anti-Lag & Exposure Stabilizers
    cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap_.set(cv::CAP_PROP_AUTO_EXPOSURE, 1); // 1 = Manual Mode in V4L2
    cap_.set(cv::CAP_PROP_EXPOSURE, 100);    // Adjust to physical environment

    return true;
}

bool AppCamera::read(cv::Mat &frame) { return cap_.read(frame); }