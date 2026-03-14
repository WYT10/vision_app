#pragma once

#include "config.h"
#include <string>

bool run_calibration(AppConfig& cfg, const std::string& config_path, std::string* error = nullptr);
