#pragma once
#include <string>
#include "params.hpp"

namespace vision_app {

bool run_probe(const AppConfig& cfg, std::string& err);
bool run_calibrate(const AppConfig& cfg, std::string& err);
bool run_deploy(const AppConfig& cfg, std::string& err);

} // namespace vision_app
