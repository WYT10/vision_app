#pragma once
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "params.hpp"

namespace vision_app {

struct ConsoleSnapshot {
    std::vector<std::string> lines;
};

class TextConsole {
public:
    explicit TextConsole(const UiConfig& cfg);
    void show(const ConsoleSnapshot& snap);
private:
    UiConfig cfg_;
    cv::Mat canvas_;
    static std::vector<std::string> wrap_line(const std::string& line,
                                              int max_width_px,
                                              double font_scale,
                                              int thickness);
};

} // namespace vision_app
