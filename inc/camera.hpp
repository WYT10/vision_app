#pragma once

#include <opencv2/videoio.hpp>

#include "types.hpp"

namespace app {

int backendFromName(const std::string& name);
std::string backendName(int backend);
cv::VideoCapture openCamera(const CameraProfile& profile);
cv::Mat readFrame(cv::VideoCapture& capture, bool flip_horizontal);

} // namespace app
