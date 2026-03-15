#pragma once

#include "config.h"
#include <string>

/*
==============================================================================
deploy.h
==============================================================================
Purpose
    Runtime trigger loop using saved calibration.

Input contract
    - calibration.valid must be true
    - requested camera mode must match calibration.camera_mode_used
    - both ROI ratios must be valid
==============================================================================
*/

bool run_deploy(const AppConfig& cfg, std::string* error = nullptr);
