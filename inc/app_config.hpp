#pragma once
#include "vision_app.hpp"

namespace vision_app
{
    bool load_config_file(const std::string &path, AppOptions &opt);
    bool parse_args(int argc, char **argv, AppOptions &opt, std::string &err);
    void print_help();
}