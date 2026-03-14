#pragma once

#include <vector>

#include "types.hpp"

namespace app {

ProbeResult probeSingleCombination(int camera_index, int backend, int width, int height, int fps);
std::vector<ProbeResult> runProbe(const AppConfig& config, const fs::path& report_dir);

} // namespace app
