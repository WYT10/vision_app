#pragma once

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace vision_app {

struct TextConsole {
    std::string window_name = "vision_app_text";
    bool enabled = true;
    int width = 880;
    int height = 420;
    double font_scale = 0.5;
    int thickness = 1;
    int line_gap = 6;
    int padding = 12;

    void open() const {
        if (!enabled) return;
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(window_name, width, height);
    }

    static std::vector<std::string> wrap_line(const std::string& line, int max_chars) {
        if (max_chars <= 8 || static_cast<int>(line.size()) <= max_chars) return {line};
        std::vector<std::string> out;
        std::istringstream iss(line);
        std::string word;
        std::string cur;
        while (iss >> word) {
            if (cur.empty()) {
                cur = word;
            } else if (static_cast<int>(cur.size() + 1 + word.size()) <= max_chars) {
                cur += ' ';
                cur += word;
            } else {
                out.push_back(cur);
                cur = word;
            }
        }
        if (!cur.empty()) out.push_back(cur);
        if (out.empty()) out.push_back(line);
        return out;
    }

    void show(const std::vector<std::string>& lines) const {
        if (!enabled) return;
        cv::Rect rect = cv::getWindowImageRect(window_name);
        int w = rect.width > 10 ? rect.width : width;
        int h = rect.height > 10 ? rect.height : height;
        cv::Mat img(h, w, CV_8UC3, cv::Scalar(18, 18, 18));
        int baseline = 0;
        cv::Size probe = cv::getTextSize("Ag", cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
        const int line_h = probe.height + line_gap;
        const int usable_w = std::max(40, w - 2 * padding);
        const int approx_chars = std::max(10, usable_w / std::max(6, probe.width / 2));
        int y = padding + probe.height;
        for (const auto& line : lines) {
            for (const auto& part : wrap_line(line, approx_chars)) {
                if (y > h - padding) break;
                cv::putText(img, part, {padding, y}, cv::FONT_HERSHEY_SIMPLEX,
                            font_scale, cv::Scalar(235, 235, 235), thickness, cv::LINE_AA);
                y += line_h;
            }
            if (y > h - padding) break;
        }
        cv::imshow(window_name, img);
    }
};

} // namespace vision_app
