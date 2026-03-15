#include "stats.hpp"

#include <iomanip>
#include <iostream>

namespace vision_app {

void print_runtime_stats(const AppOptions& opt, const RuntimeStats& s) {
    std::cout << "=== Test Result ===\n";
    std::cout << "Device            : " << opt.device << "\n";
    std::cout << "Requested         : " << opt.width << "x" << opt.height << " @ " << opt.fps << " fps " << opt.fourcc << "\n";
    std::cout << "Actual            : " << s.actual_width << "x" << s.actual_height << "\n";
    std::cout << "IO mode           : " << (opt.io_mode == IoMode::Grab ? "grab+retrieve" : "read") << "\n";
    std::cout << "Latest only       : " << (opt.latest_only ? "yes" : "no") << "\n";
    std::cout << "Drain grabs       : " << opt.drain_grabs << "\n";
    std::cout << "Frames            : " << s.frames << "\n";
    std::cout << "Elapsed sec       : " << s.elapsed_sec << "\n\n";

    std::cout << "Capture Performance\n";
    std::cout << "  Avg FPS         : " << s.fps_avg << "\n";
    std::cout << "  Min FPS         : " << s.fps_min << "\n";
    std::cout << "  Max FPS         : " << s.fps_max << "\n";
    std::cout << "  Avg Frame ms    : " << s.frame_time_avg_ms << "\n";
    std::cout << "  Min Frame ms    : " << s.frame_time_min_ms << "\n";
    std::cout << "  Max Frame ms    : " << s.frame_time_max_ms << "\n\n";

    std::cout << "Queue / Freshness\n";
    std::cout << "  Empty frames    : " << s.empty_frames << "\n";
    std::cout << "  Stale discarded : " << s.stale_grabs_discarded << "\n";
    std::cout << "  Buffer set ok   : " << (s.backend_buffer_request_ok ? "yes" : "no") << "\n";
    std::cout << "  Buffer after set: " << s.backend_buffer_size_after_set << "\n\n";

    std::cout << "Assessment\n";
    std::cout << "  Target ratio    : " << (s.target_ratio * 100.0) << "%\n";
    std::cout << "  Target met      : " << (s.target_met ? "yes" : "no") << "\n\n";
}

} // namespace vision_app
