#pragma once
#include "vision_app.hpp"

namespace vision_app {
bool ensure_report_dirs(const AppOptions& opt, std::string& err);
bool append_test_csv(const std::string& path, const AppOptions& opt, const RuntimeStats& stats);
bool write_markdown_report(const std::string& path, const AppOptions& opt, const ProbeResult& probe, const RuntimeStats& stats);
} // namespace vision_app
