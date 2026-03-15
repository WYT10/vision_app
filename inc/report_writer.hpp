#pragma once
#include "vision_app.hpp"

namespace vision_app {
bool ensure_parent_dir(const std::string& path);
bool write_markdown_report(const std::string& path, const AppOptions& opt, const ProbeResult& probe, const RuntimeStats& stats);
} // namespace vision_app
