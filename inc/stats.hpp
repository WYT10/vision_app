#pragma once
#include "vision_app.hpp"

namespace vision_app
{
    bool write_stats_csv(const std::string &path, const AppOptions &opt, const RuntimeStats &stats);
    void print_runtime_stats(const AppOptions &opt, const RuntimeStats &stats);
}