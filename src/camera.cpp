#include "camera.hpp"

#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "utils.hpp"

namespace app {

int backendFromName(const std::string& name) {
    const std::string backend = lower(name);
    if (backend == "any") return cv::CAP_ANY;
#ifdef _WIN32
    if (backend == "msmf") return cv::CAP_MSMF;
    if (backend == "dshow") return cv::CAP_DSHOW;
#else
    if (backend == "v4l2") return cv::CAP_V4L2;
    if (backend == "gstreamer") return cv::CAP_GSTREAMER;
#endif
    return cv::CAP_ANY;
}

std::string backendName(int backend) {
#ifdef _WIN32
    if (backend == cv::CAP_MSMF) return "MSMF";
    if (backend == cv::CAP_DSHOW) return "DSHOW";
#endif
#ifdef __linux__
    if (backend == cv::CAP_V4L2) return "V4L2";
    if (backend == cv::CAP_GSTREAMER) return "GSTREAMER";
#endif
    return "ANY";
}

cv::VideoCapture openCamera(const CameraProfile& profile) {
    cv::VideoCapture capture(profile.index, backendFromName(profile.backend));
    if (!capture.isOpened()) {
        throw std::runtime_error("Failed to open camera index " + std::to_string(profile.index));
    }

    capture.set(cv::CAP_PROP_FRAME_WIDTH, profile.width);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, profile.height);
    capture.set(cv::CAP_PROP_FPS, profile.fps);
    if (profile.use_mjpg) {
        capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    }

    cv::Mat throwaway;
    for (int i = 0; i < profile.warmup_frames; ++i) {
        capture >> throwaway;
    }

    return capture;
}

cv::Mat readFrame(cv::VideoCapture& capture, bool flip_horizontal) {
    cv::Mat frame;
    capture >> frame;
    if (frame.empty()) {
        return frame;
    }
    if (flip_horizontal) {
        cv::flip(frame, frame, 1);
    }
    return frame;
}

} // namespace app
