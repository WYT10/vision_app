#pragma once

#include "config.h"
#include <string>

bool run_deploy(const AppConfig& cfg, std::string* error = nullptr);
