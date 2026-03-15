#include "report_writer.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace vision_app {

static void ensure_parent(const std::string& path, std::string& err) {
    const std::filesystem::path p(path);
    const auto parent = p.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) err = ec.message();
    }
}

bool ensure_report_dirs(const AppOptions& opt, std::string& err) {
    err.clear();
    if (!opt.report_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(opt.report_dir, ec);
        if (ec) {
            err = ec.message();
            return false;
        }
    }
    ensure_parent(opt.csv_path, err);
    if (!err.empty()) return false;
    ensure_parent(opt.probe_csv_path, err);
    if (!err.empty()) return false;
    ensure_parent(opt.markdown_path, err);
    return err.empty();
}

static std::string now_timestamp() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const std::time_t t = system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

bool append_test_csv(const std::string& path, const AppOptions& opt, const RuntimeStats& s) {
    bool need_header = false;
    {
        std::ifstream test(path);
        need_header = !test.good() || test.peek() == std::ifstream::traits_type::eof();
    }

    std::ofstream out(path, std::ios::out | std::ios::app);
    if (!out.is_open()) return false;

    if (need_header) {
        out << "timestamp,device,format,width_req,height_req,fps_req,width_actual,height_actual,duration_sec,frames,"
               "fps_avg,fps_min,fps_max,frame_ms_avg,frame_ms_min,frame_ms_max,empty_frames,stale_grabs_discarded,"
               "latest_only,io_mode,buffer_size_after_set,target_ratio,target_met\n";
    }

    out << now_timestamp() << ','
        << '"' << opt.device << "\","
        << '"' << opt.fourcc << "\"," 
        << opt.width << ',' << opt.height << ',' << opt.fps << ','
        << s.actual_width << ',' << s.actual_height << ','
        << s.elapsed_sec << ',' << s.frames << ','
        << s.fps_avg << ',' << s.fps_min << ',' << s.fps_max << ','
        << s.frame_time_avg_ms << ',' << s.frame_time_min_ms << ',' << s.frame_time_max_ms << ','
        << s.empty_frames << ',' << s.stale_grabs_discarded << ','
        << (opt.latest_only ? 1 : 0) << ','
        << '"' << (opt.io_mode == IoMode::Grab ? "grab" : "read") << "\","
        << s.backend_buffer_size_after_set << ','
        << s.target_ratio << ',' << (s.target_met ? 1 : 0) << '\n';

    return true;
}

bool write_markdown_report(const std::string& path, const AppOptions& opt, const ProbeResult& probe, const RuntimeStats& s) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) return false;

    out << "# Vision App Report\n\n";
    out << "## Device\n";
    out << "- Device: `" << probe.device << "`\n";
    out << "- Card: `" << probe.card_name << "`\n";
    out << "- Bus: `" << probe.bus_info << "`\n\n";

    out << "## Request\n";
    out << "- Format: `" << opt.fourcc << "`\n";
    out << "- Resolution: `" << opt.width << "x" << opt.height << "`\n";
    out << "- Target FPS: `" << opt.fps << "`\n";
    out << "- IO mode: `" << (opt.io_mode == IoMode::Grab ? "grab" : "read") << "`\n";
    out << "- Latest only: `" << (opt.latest_only ? "true" : "false") << "`\n";
    out << "- Drain grabs: `" << opt.drain_grabs << "`\n\n";

    out << "## Result\n";
    out << "- Actual frame size: `" << s.actual_width << "x" << s.actual_height << "`\n";
    out << "- Frames: `" << s.frames << "`\n";
    out << "- Average FPS: `" << s.fps_avg << "`\n";
    out << "- Min FPS: `" << s.fps_min << "`\n";
    out << "- Max FPS: `" << s.fps_max << "`\n";
    out << "- Average frame time ms: `" << s.frame_time_avg_ms << "`\n";
    out << "- Empty frames: `" << s.empty_frames << "`\n";
    out << "- Discarded stale grabs: `" << s.stale_grabs_discarded << "`\n";
    out << "- Target ratio: `" << s.target_ratio << "`\n";
    out << "- Target met: `" << (s.target_met ? "yes" : "no") << "`\n";
    out << "- Requested buffer set ok: `" << (s.backend_buffer_request_ok ? "yes" : "no") << "`\n";
    out << "- Buffer size after set: `" << s.backend_buffer_size_after_set << "`\n\n";

    out << "## Probe table\n\n";
    out << "| Format | Resolution | FPS |\n";
    out << "|---|---:|---|\n";
    for (const auto& m : probe.modes) {
        std::ostringstream fpss;
        for (size_t i = 0; i < m.fps_list.size(); ++i) {
            if (i) fpss << ',';
            fpss << m.fps_list[i];
        }
        out << "| " << m.pixel_format << " | " << m.width << 'x' << m.height << " | " << fpss.str() << " |\n";
    }

    return true;
}

} // namespace vision_app
