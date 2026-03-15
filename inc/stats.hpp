#pragma once
#include "vision_app.hpp"

namespace vision_app {
void print_runtime_stats(const AppOptions& opt, const RuntimeStats& stats);
bool write_stats_csv(const std::string& path, const AppOptions& opt, const RuntimeStats& stats);
} // namespace vision_app
