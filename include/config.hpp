#pragma once
#include <string>
#include "params.hpp"

namespace vision_app {

bool load_default_config(AppConfig& cfg, std::string& err);
bool load_config_ini(const std::string& path, AppConfig& cfg, std::string& err);
bool apply_cli_overrides(int argc, char** argv, AppConfig& cfg, std::string& err);
bool load_config_from_argv(int argc, char** argv, AppConfig& cfg, std::string& err);
std::string dump_effective_config(const AppConfig& cfg);

} // namespace vision_app
