#include "report_writer.hpp"

#include <filesystem>
#include <fstream>

namespace vision_app {

bool ensure_parent_dir(const std::string& path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path p(path);
    const fs::path parent = p.parent_path();
    if (parent.empty()) return true;
    fs::create_directories(parent, ec);
    return !ec;
}

bool write_markdown_report(const std::string& path, const AppOptions& opt, const ProbeResult& probe, const RuntimeStats& stats) {
    if (!ensure_parent_dir(path)) return false;

    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) return false;

    out << "# Vision App Camera Report\n\n";
    out << "## Device Info\n";
    out << "- Device: `" << probe.device << "`\n";
    out << "- Card: `" << probe.card_name << "`\n";
    out << "- Bus: `" << probe.bus_info << "`\n\n";

    out << "## Requested Test\n";
    out << "- Format: `" << opt.fourcc << "`\n";
    out << "- Resolution: `" << opt.width << 'x' << opt.height << "`\n";
    out << "- Target FPS: `" << opt.fps << "`\n";
    out << "- Duration: `" << stats.elapsed_sec << " s`\n\n";

    out << "## Measured Result\n";
    out << "- Actual Resolution: `" << stats.width << 'x' << stats.height << "`\n";
    out << "- Frames: `" << stats.frames << "`\n";
    out << "- Average FPS: `" << stats.fps_avg << "`\n";
    out << "- Min FPS: `" << stats.fps_min << "`\n";
    out << "- Max FPS: `" << stats.fps_max << "`\n";
    out << "- Average frame time: `" << stats.frame_time_avg_ms << " ms`\n";
    out << "- Jitter stddev: `" << stats.frame_time_stddev_ms << " ms`\n\n";

    out << "## Assessment\n";
    out << "- Reported by camera: `" << (stats.mode_reported ? "yes" : "no") << "`\n";
    out << "- Target met: `" << (stats.target_met ? "yes" : "no") << "`\n";
    out << "- Achieved ratio: `" << (stats.target_ratio * 100.0) << "%`\n";
    out << "- Stability: `" << stats.stability << "`\n\n";

    out << "## Advertised Modes Snapshot\n\n";
    out << "| Format | Resolution | Supported FPS |\n";
    out << "|---|---|---|\n";
    for (const auto& m : probe.modes) {
        out << '|' << m.pixel_format << '|' << m.width << 'x' << m.height << '|';
        for (size_t i = 0; i < m.fps_list.size(); ++i) {
            if (i) out << ',';
            if (m.fps_list[i] == static_cast<int>(m.fps_list[i])) out << static_cast<int>(m.fps_list[i]);
            else out << m.fps_list[i];
        }
        out << "|\n";
    }

    return true;
}

} // namespace vision_app
