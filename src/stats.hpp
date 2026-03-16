#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace vision_app {

struct BenchStats {
    unsigned long long frames = 0;
    double elapsed_sec = 0.0;
    double fps_avg = 0.0;
    double fps_min = 0.0;
    double fps_max = 0.0;
    double frame_ms_avg = 0.0;
    double frame_ms_min = 0.0;
    double frame_ms_max = 0.0;
    int actual_w = 0;
    int actual_h = 0;
};

static inline void print_bench_stats(const BenchStats& s) {
    std::cout << "\n=== Runtime stats ===\n";
    std::cout << "Actual frame size : " << s.actual_w << "x" << s.actual_h << "\n";
    std::cout << "Frames            : " << s.frames << "\n";
    std::cout << "Elapsed sec       : " << s.elapsed_sec << "\n";
    std::cout << "FPS avg           : " << s.fps_avg << "\n";
    std::cout << "FPS min           : " << s.fps_min << "\n";
    std::cout << "FPS max           : " << s.fps_max << "\n";
    std::cout << "Frame time avg ms : " << s.frame_ms_avg << "\n";
    std::cout << "Frame time min ms : " << s.frame_ms_min << "\n";
    std::cout << "Frame time max ms : " << s.frame_ms_max << "\n";
}

static inline bool append_bench_csv(const std::string& path,
                                    const std::string& device,
                                    int req_w, int req_h, int req_fps,
                                    const std::string& fourcc,
                                    const BenchStats& s) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    const bool exists = std::filesystem::exists(path);
    std::ofstream out(path, std::ios::app);
    if (!out.is_open()) return false;
    if (!exists) {
        out << "device,width_req,height_req,fps_req,fourcc,width_actual,height_actual,frames,elapsed_sec,fps_avg,fps_min,fps_max,frame_ms_avg,frame_ms_min,frame_ms_max\n";
    }
    out << device << ',' << req_w << ',' << req_h << ',' << req_fps << ',' << fourcc << ','
        << s.actual_w << ',' << s.actual_h << ',' << s.frames << ',' << s.elapsed_sec << ','
        << s.fps_avg << ',' << s.fps_min << ',' << s.fps_max << ','
        << s.frame_ms_avg << ',' << s.frame_ms_min << ',' << s.frame_ms_max << '\n';
    return true;
}

static inline bool write_latest_report_md(const std::string& path,
                                          const std::string& mode,
                                          const std::string& body) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path, std::ios::trunc);
    if (!out.is_open()) return false;
    out << "# vision_app report\n\n";
    out << "## Mode\n- " << mode << "\n\n";
    out << body << '\n';
    return true;
}

} // namespace vision_app
