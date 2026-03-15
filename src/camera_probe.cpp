#include "camera_probe.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <iomanip>
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
    while (fgets(buf.data(), static_cast<int>(buf.size()), fp) != nullptr) {
        out += buf.data();
    }
    const int rc = pclose(fp);
    return rc == 0;
}

static void normalize_modes(ProbeResult& out) {
    std::map<std::tuple<std::string, uint32_t, uint32_t>, std::set<double>> merged;
    for (const auto& m : out.modes) {
        auto& fps = merged[std::make_tuple(m.pixel_format, m.width, m.height)];
        for (double v : m.fps_list) fps.insert(v);
    }

    std::vector<CameraMode> normalized;
    normalized.reserve(merged.size());
    for (const auto& kv : merged) {
        CameraMode m;
        m.pixel_format = std::get<0>(kv.first);
        m.width = std::get<1>(kv.first);
        m.height = std::get<2>(kv.first);
        m.fps_list.assign(kv.second.begin(), kv.second.end());
        normalized.push_back(std::move(m));
    }

    std::sort(normalized.begin(), normalized.end(), [](const CameraMode& a, const CameraMode& b) {
        if (a.pixel_format != b.pixel_format) return a.pixel_format < b.pixel_format;
        if (a.width != b.width) return a.width < b.width;
        return a.height < b.height;
    });
    out.modes.swap(normalized);
}

bool probe_camera_modes(const std::string& device, ProbeResult& out, std::string& err) {
    out = {};
    out.device = device;

    std::string info;
    run_command("v4l2-ctl -d " + device + " --all 2>/dev/null", info);

    {
        std::regex card_re(R"(Card type\s*:\s*(.+))");
        std::regex bus_re(R"(Bus info\s*:\s*(.+))");
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
    CameraMode current;
    bool have_format = false;

    std::regex pix_re(R"(\[\d+\]:\s+'([^']+)')");
    std::regex size_re(R"(Size:\s+Discrete\s+(\d+)x(\d+))");
    std::regex fps_re(R"(([\d.]+)\s+fps)");
    std::smatch m;

    while (std::getline(iss, line)) {
        if (std::regex_search(line, m, pix_re)) {
            if (current.width > 0) out.modes.push_back(current);
            current = {};
            current.pixel_format = m[1];
            have_format = true;
        } else if (std::regex_search(line, m, size_re)) {
            if (current.width > 0) out.modes.push_back(current);
            current.width = static_cast<uint32_t>(std::stoul(m[1]));
            current.height = static_cast<uint32_t>(std::stoul(m[2]));
        } else if (have_format && std::regex_search(line, m, fps_re) && current.width > 0) {
            current.fps_list.push_back(std::stod(m[1]));
        }
    }
    if (current.width > 0) out.modes.push_back(current);

    normalize_modes(out);

    if (out.modes.empty()) {
        err = "no camera modes parsed from v4l2-ctl output";
        return false;
    }
    return true;
}

void print_probe_result(const ProbeResult& probe) {
    std::cout << "\n=== Camera Probe Summary ===\n";
    std::cout << "Device      : " << probe.device << "\n";
    std::cout << "Card        : " << probe.card_name << "\n";
    std::cout << "Bus         : " << probe.bus_info << "\n";
    std::cout << "Modes       : " << probe.modes.size() << " unique format/resolution entries\n\n";

    std::cout << std::left
              << std::setw(8)  << "Format"
              << std::setw(12) << "Resolution"
              << "FPS\n";
    std::cout << std::string(40, '-') << "\n";

    for (const auto& m : probe.modes) {
        std::ostringstream fpss;
        for (size_t i = 0; i < m.fps_list.size(); ++i) {
            if (i) fpss << ',';
            fpss << m.fps_list[i];
        }
        std::ostringstream res;
        res << m.width << 'x' << m.height;
        std::cout << std::setw(8) << m.pixel_format
                  << std::setw(12) << res.str()
                  << fpss.str() << "\n";
    }
    std::cout << "\n";
}

bool write_probe_csv(const std::string& path, const ProbeResult& probe) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) return false;

    out << "device,card,bus,pixel_format,width,height,supported_fps\n";
    for (const auto& m : probe.modes) {
        std::ostringstream fpss;
        for (size_t i = 0; i < m.fps_list.size(); ++i) {
            if (i) fpss << ',';
            fpss << m.fps_list[i];
        }
        out << '"' << probe.device << "\"," << '"' << probe.card_name << "\"," << '"' << probe.bus_info << "\",";
        out << '"' << m.pixel_format << "\"," << m.width << ',' << m.height << ',' << '"' << fpss.str() << "\"\n";
    }
    return true;
}

} // namespace vision_app
