#pragma once

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "camera.hpp"

namespace vision_app {

inline void print_runtime_stats(const RuntimeStats& s) {
    std::cout << "\n=== Runtime stats ===\n";
    std::cout << "Actual frame size : " << s.actual_width << 'x' << s.actual_height << "\n";
    std::cout << "Frames            : " << s.frames << "\n";
    std::cout << "Elapsed sec       : " << s.elapsed_sec << "\n";
    std::cout << "FPS avg           : " << s.fps_avg << "\n";
    std::cout << "FPS min           : " << s.fps_min << "\n";
    std::cout << "FPS max           : " << s.fps_max << "\n";
    std::cout << "Frame time avg ms : " << s.frame_time_avg_ms << "\n";
    std::cout << "Frame time min ms : " << s.frame_time_min_ms << "\n";
    std::cout << "Frame time max ms : " << s.frame_time_max_ms << "\n";
}

inline bool write_report_md(const std::string& path,
                            const std::string& title,
                            const RuntimeStats* s,
                            const std::string& extra) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << "# " << title << "\n\n";
    if (s) {
        out << "- Actual frame size: " << s->actual_width << "x" << s->actual_height << "\n";
        out << "- Frames: " << s->frames << "\n";
        out << "- Elapsed sec: " << s->elapsed_sec << "\n";
        out << "- FPS avg: " << s->fps_avg << "\n";
        out << "- FPS min: " << s->fps_min << "\n";
        out << "- FPS max: " << s->fps_max << "\n\n";
    }
    out << extra << "\n";
    return true;
}

} // namespace vision_app
