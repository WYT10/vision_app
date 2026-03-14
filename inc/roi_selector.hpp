#pragma once

#include "types.hpp"

namespace app {

void beginSelection(SelectionState& state, RoiNorm* target, int ref_w, int ref_h, const std::string& label);
void attachSelectionMouseCallback(const std::string& window_name, SelectionState* state);
void drawSelectionOverlay(cv::Mat& image, const SelectionState& state);

} // namespace app
