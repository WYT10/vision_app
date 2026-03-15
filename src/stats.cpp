#include "stats.hpp"
#include "report_writer.hpp"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace vision_app {

static std::string iso_timestamp_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

void print_runtime_stats(const AppOptions& opt, const RuntimeStats& s) {
    std::cout << "\n=== Test Result ===\n";
    std::cout << "Device            : " << opt.device << "\n";
    std::cout << "Requested         : " << opt.width << 'x' << opt.height << " @ " << opt.fps << " fps " << opt.fourcc << "\n";
    std::cout << "Actual            : " << s.width << 'x' << s.height << "\n";
    std::cout << "Duration          : " << s.elapsed_sec << " s\n";
    std::cout << "Frames            : " << s.frames << "\n\n";

    std::cout << "Capture Performance\n";
    std::cout << "  Avg FPS         : " << s.fps_avg << "\n";
    std::cout << "  Min FPS         : " << s.fps_min << "\n";
    std::cout << "  Max FPS         : " << s.fps_max << "\n";
    std::cout << "  Avg Frame ms    : " << s.frame_time_avg_ms << "\n";
    std::cout << "  Min Frame ms    : " << s.frame_time_min_ms << "\n";
    std::cout << "  Max Frame ms    : " << s.frame_time_max_ms << "\n";
    std::cout << "  Jitter stddev   : " << s.frame_time_stddev_ms << " ms\n\n";

    std::cout << "Quality Assessment\n";
    std::cout << "  Reported by cam : " << (s.mode_reported ? "yes" : "no") << "\n";
    std::cout << "  Target met      : " << (s.target_met ? "yes" : "no") << "\n";
    std::cout << "  Target ratio    : " << (100.0 * s.target_ratio) << "%\n";
    std::cout << "  Stability       : " << s.stability << "\n";
}

bool write_stats_csv(const std::string& path, const AppOptions& opt, const RuntimeStats& s) {
    if (!ensure_parent_dir(path)) return false;

    bool new_file = false;
    {
        std::ifstream in(path);
        new_file = !in.good() || in.peek() == std::ifstream::traits_type::eof();
    }

    std::ofstream out(path, std::ios::out | std::ios::app);
    if (!out.is_open()) return false;

    if (new_file) {
        out << "timestamp,device,format,width_req,height_req,fps_req,width_actual,height_actual,duration_sec,frames,read_failures,fps_avg,fps_min,fps_max,frame_ms_avg,frame_ms_min,frame_ms_max,frame_ms_stddev,target_ratio,target_met,mode_reported,stability\n";
    }

    out << iso_timestamp_now() << ','
        << '"' << opt.device << "\"," << opt.fourcc << ','
        << opt.width << ',' << opt.height << ',' << opt.fps << ','
        << s.width << ',' << s.height << ','
        << s.elapsed_sec << ',' << s.frames << ',' << s.read_failures << ','
        << s.fps_avg << ',' << s.fps_min << ',' << s.fps_max << ','
        << s.frame_time_avg_ms << ',' << s.frame_time_min_ms << ',' << s.frame_time_max_ms << ','
        << s.frame_time_stddev_ms << ',' << s.target_ratio << ','
        << (s.target_met ? 1 : 0) << ',' << (s.mode_reported ? 1 : 0) << ',' << s.stability << '\n';

    return true;
}

} // namespace vision_app
