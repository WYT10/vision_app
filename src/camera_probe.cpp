#include "camera_probe.hpp"
#include "report_writer.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <tuple>

namespace vision_app {

static bool run_command(const std::string& cmd, std::string& out) {
    out.clear();
    std::array<char, 512> buf{};
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) return false;

    while (fgets(buf.data(), static_cast<int>(buf.size()), fp)) {
        out += buf.data();
    }

    return pclose(fp) == 0;
}

static std::vector<CameraMode> normalize_modes(const std::vector<CameraMode>& raw) {
    std::map<std::tuple<std::string, uint32_t, uint32_t>, std::set<double>> grouped;

    for (const auto& mode : raw) {
        if (mode.width == 0 || mode.height == 0 || mode.pixel_format.empty()) continue;
        auto& fps_set = grouped[std::make_tuple(mode.pixel_format, mode.width, mode.height)];
        for (double fps : mode.fps_list) {
            if (fps > 0.0) fps_set.insert(fps);
        }
    }

    std::vector<CameraMode> out;
    out.reserve(grouped.size());
    for (const auto& it : grouped) {
        CameraMode m;
        m.pixel_format = std::get<0>(it.first);
        m.width = std::get<1>(it.first);
        m.height = std::get<2>(it.first);
        m.fps_list.assign(it.second.begin(), it.second.end());
        out.push_back(std::move(m));
    }

    std::sort(out.begin(), out.end(), [](const CameraMode& a, const CameraMode& b) {
        if (a.pixel_format != b.pixel_format) return a.pixel_format < b.pixel_format;
        if (a.width != b.width) return a.width < b.width;
        return a.height < b.height;
    });
    return out;
}

bool probe_camera_modes(const std::string& device, ProbeResult& out, std::string& err) {
    out = {};
    out.device = device;

    std::string info;
    run_command("v4l2-ctl -d " + device + " --all 2>/dev/null", info);

    {
        const std::regex card_re(R"(Card type\s*:\s*(.+))");
        const std::regex bus_re(R"(Bus info\s*:\s*(.+))");
        std::smatch m;
        std::istringstream iss(info);
        std::string line;
        while (std::getline(iss, line)) {
            if (std::regex_search(line, m, card_re)) out.card_name = m[1];
            if (std::regex_search(line, m, bus_re)) out.bus_info = m[1];
        }
    }

    std::string formats;
    if (!run_command("v4l2-ctl -d " + device + " --list-formats-ext 2>/dev/null", formats)) {
        err = "failed to run v4l2-ctl --list-formats-ext";
        return false;
    }

    std::istringstream iss(formats);
    std::string line;
    std::string current_pixel;
    uint32_t current_w = 0;
    uint32_t current_h = 0;
    std::vector<CameraMode> raw_modes;

    const std::regex pix_re(R"(\[\d+\]:\s+'([^']+)')");
    const std::regex size_re(R"(Size:\s+Discrete\s+(\d+)x(\d+))");
    const std::regex fps_re(R"((\d+(?:\.\d+)?)\s+fps)");
    std::smatch m;

    while (std::getline(iss, line)) {
        if (std::regex_search(line, m, pix_re)) {
            current_pixel = m[1];
            current_w = 0;
            current_h = 0;
            continue;
        }
        if (std::regex_search(line, m, size_re)) {
            current_w = static_cast<uint32_t>(std::stoul(m[1]));
            current_h = static_cast<uint32_t>(std::stoul(m[2]));
            raw_modes.push_back(CameraMode{current_pixel, current_w, current_h, {}});
            continue;
        }
        if (std::regex_search(line, m, fps_re) && !raw_modes.empty() && current_w > 0 && current_h > 0) {
            raw_modes.back().fps_list.push_back(std::stod(m[1]));
        }
    }

    out.modes = normalize_modes(raw_modes);
    if (out.modes.empty()) {
        err = "no camera modes parsed from v4l2-ctl output";
        return false;
    }
    return true;
}

bool probe_has_mode(const ProbeResult& probe, const AppOptions& opt) {
    for (const auto& m : probe.modes) {
        if (m.pixel_format == opt.fourcc && static_cast<int>(m.width) == opt.width && static_cast<int>(m.height) == opt.height) {
            for (double fps : m.fps_list) {
                if (static_cast<int>(fps + 0.5) == opt.fps) return true;
            }
        }
    }
    return false;
}

void print_probe_result(const ProbeResult& probe) {
    std::cout << "=== Camera Probe Summary ===\n";
    std::cout << "Device      : " << probe.device << "\n";
    std::cout << "Card        : " << probe.card_name << "\n";
    std::cout << "Bus         : " << probe.bus_info << "\n";
    std::cout << "Modes       : " << probe.modes.size() << " unique format-resolution entries\n\n";

    const int w_fmt = 8;
    const int w_res = 11;
    const int w_fps = 18;

    auto hr = [&]() {
        std::cout << '+' << std::string(w_fmt + 2, '-')
                  << '+' << std::string(w_res + 2, '-')
                  << '+' << std::string(w_fps + 2, '-') << "+\n";
    };

    hr();
    std::cout << "| " << std::left << std::setw(w_fmt) << "Format"
              << " | " << std::left << std::setw(w_res) << "Resolution"
              << " | " << std::left << std::setw(w_fps) << "Supported FPS" << " |\n";
    hr();

    for (const auto& m : probe.modes) {
        std::ostringstream fpss;
        for (size_t i = 0; i < m.fps_list.size(); ++i) {
            if (i) fpss << ',';
            if (m.fps_list[i] == static_cast<int>(m.fps_list[i])) fpss << static_cast<int>(m.fps_list[i]);
            else fpss << std::fixed << std::setprecision(2) << m.fps_list[i];
        }

        std::ostringstream res;
        res << m.width << 'x' << m.height;

        std::cout << "| " << std::left << std::setw(w_fmt) << m.pixel_format
                  << " | " << std::left << std::setw(w_res) << res.str()
                  << " | " << std::left << std::setw(w_fps) << fpss.str() << " |\n";
    }
    hr();
}

bool write_probe_csv(const std::string& path, const ProbeResult& probe) {
    if (!ensure_parent_dir(path)) return false;

    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) return false;

    out << "device,card,bus,pixel_format,width,height,supported_fps\n";
    for (const auto& m : probe.modes) {
        std::ostringstream fpss;
        for (size_t i = 0; i < m.fps_list.size(); ++i) {
            if (i) fpss << ',';
            if (m.fps_list[i] == static_cast<int>(m.fps_list[i])) fpss << static_cast<int>(m.fps_list[i]);
            else fpss << m.fps_list[i];
        }
        out << '"' << probe.device << "\",\""
            << probe.card_name << "\",\""
            << probe.bus_info << "\","
            << m.pixel_format << ',' << m.width << ',' << m.height << ",\""
            << fpss.str() << "\"\n";
    }
    return true;
}

} // namespace vision_app
