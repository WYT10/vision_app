#pragma once
#include "vision_app.hpp"

namespace vision_app {
bool run_camera_test(const AppOptions& opt, const ProbeResult& probe, RuntimeStats& stats, std::string& err);
} // namespace vision_app
