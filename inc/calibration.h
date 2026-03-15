#pragma once

#include "config.h"
#include <string>

/*
==============================================================================
calibration.h
==============================================================================
Purpose
    Live calibration loop.

Workflow
    1. Open camera using the single active camera mode.
    2. Detect AprilTag in raw frame.
    3. Preview homography warp.
    4. Lock transform.
    5. Select red ROI + image ROI on warped frame.
    6. Save calibration payload back into config JSON.
==============================================================================
*/

bool run_calibration(AppConfig& cfg, const std::string& config_path, std::string* error = nullptr);
