#include "text_console.hpp"
#include <algorithm>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace vision_app {

TextConsole::TextConsole(const UiConfig& cfg) : cfg_(cfg) {}

std::vector<std::string> TextConsole::wrap_line(const std::string& line,
                                                int max_width_px,
                                                double font_scale,
                                                int thickness) {
    std::vector<std::string> out;
    if (line.empty()) {
        out.push_back("");
        return out;
    }
    std::string current;
    std::string word;
    auto flush_word = [&]() {
        if (word.empty()) return;
        std::string candidate = current.empty() ? word : current + " " + word;
        int baseline = 0;
        cv::Size ts = cv::getTextSize(candidate, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
        if (ts.width <= max_width_px || current.empty()) {
            current = candidate;
        } else {
            out.push_back(current);
            current = word;
        }
        word.clear();
    };

    for (char ch : line) {
        if (ch == ' ') {
            flush_word();
        } else {
            word.push_back(ch);
        }
    }
    flush_word();
    if (!current.empty()) out.push_back(current);
    if (out.empty()) out.push_back(line);
    return out;
}

void TextConsole::show(const ConsoleSnapshot& snap) {
    const std::string win = "vision_app_text";
    cv::namedWindow(win, cv::WINDOW_NORMAL);

    int width = 900;
    int height = 560;
    try {
        const cv::Rect r = cv::getWindowImageRect(win);
        if (r.width > 64 && r.height > 64) {
            width = r.width;
            height = r.height;
        }
    } catch (...) {
        // fallback to defaults if backend does not support querying the window image rect.
    }

    canvas_ = cv::Mat(height, width, CV_8UC3, cv::Scalar(28, 28, 28));

    const int pad = std::max(4, cfg_.text_console_padding);
    const int usable_w = std::max(50, width - 2 * pad);
    const double fs = std::max(0.3, cfg_.text_console_font_scale);
    const int thickness = 1;
    const int line_h = std::max(16, static_cast<int>(std::round(24 * fs)));

    int y = pad + line_h;
    std::vector<std::string> wrapped;
    for (const auto& line : snap.lines) {
        auto parts = wrap_line(line, usable_w, fs, thickness);
        wrapped.insert(wrapped.end(), parts.begin(), parts.end());
    }

    for (const auto& line : wrapped) {
        if (y > height - pad) break;
        cv::putText(canvas_, line, cv::Point(pad, y), cv::FONT_HERSHEY_SIMPLEX, fs,
                    cv::Scalar(235, 235, 235), thickness, cv::LINE_AA);
        y += line_h;
    }

    cv::imshow(win, canvas_);
}

} // namespace vision_app
