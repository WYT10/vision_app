#include "stats.hpp"

#include <fstream>
#include <iostream>

namespace vision_app
{

    void print_runtime_stats(const AppOptions &opt, const RuntimeStats &s)
    {
        std::cout << "\n=== Runtime stats ===\n";
        std::cout << "Device            : " << opt.device << "\n";
        std::cout << "Requested mode    : " << opt.width << "x" << opt.height
                  << " @ " << opt.fps << " fps, " << opt.fourcc << "\n";
        std::cout << "Actual frame size : " << s.width << "x" << s.height << "\n";
        std::cout << "Frames            : " << s.frames << "\n";
        std::cout << "Elapsed sec       : " << s.elapsed_sec << "\n";
        std::cout << "FPS avg           : " << s.fps_avg << "\n";
        std::cout << "FPS min           : " << s.fps_min << "\n";
        std::cout << "FPS max           : " << s.fps_max << "\n";
        std::cout << "Frame time avg ms : " << s.frame_time_avg_ms << "\n";
        std::cout << "Frame time min ms : " << s.frame_time_min_ms << "\n";
        std::cout << "Frame time max ms : " << s.frame_time_max_ms << "\n";
    }

    bool write_stats_csv(const std::string &path, const AppOptions &opt, const RuntimeStats &s)
    {
        std::ofstream out(path, std::ios::out | std::ios::trunc);
        if (!out.is_open())
            return false;

        out << "device,width_req,height_req,fps_req,fourcc,width_actual,height_actual,frames,elapsed_sec,"
               "fps_avg,fps_min,fps_max,frame_time_avg_ms,frame_time_min_ms,frame_time_max_ms\n";

        out << opt.device << ","
            << opt.width << ","
            << opt.height << ","
            << opt.fps << ","
            << opt.fourcc << ","
            << s.width << ","
            << s.height << ","
            << s.frames << ","
            << s.elapsed_sec << ","
            << s.fps_avg << ","
            << s.fps_min << ","
            << s.fps_max << ","
            << s.frame_time_avg_ms << ","
            << s.frame_time_min_ms << ","
            << s.frame_time_max_ms << "\n";

        return true;
    }

} // namespace vision_app