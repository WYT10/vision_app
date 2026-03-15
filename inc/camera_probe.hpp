#pragma once
#include "vision_app.hpp"

namespace vision_app
{
    bool probe_camera_modes(const std::string &device, ProbeResult &out, std::string &err);
    void print_probe_result(const ProbeResult &probe);
}