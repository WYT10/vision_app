#pragma once

#include <string>

#include "app_types.hpp"

namespace vision_app {

bool load_profile_config(const std::string& path, AppOptions& opt, std::string& err);
bool save_profile_config(const std::string& path, const AppOptions& opt, std::string& err);
void resolve_profile_paths(const std::string& config_path, AppOptions& opt);
void print_help();

} // namespace vision_app
