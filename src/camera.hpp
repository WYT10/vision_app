#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

namespace vision_app {

struct CameraMode {
    std::string pixel_format;
    int width = 0;
    int height = 0;
    std::vector<double> fps_list;
};

struct ProbeResult {
    std::string device;
    std::string card_name;
    std::string bus_info;
    std::vector<CameraMode> modes;
};

struct CameraConfig {
    std::string device = "/dev/video0";
    int width = 1280;
    int height = 720;
    int fps = 30;
    std::string fourcc = "MJPG";
    int buffer_size = 1;
    bool latest_only = true;
    int drain_grabs = 2;
    int warmup_frames = 8;
    int duration_sec = 10;
    bool preview = true;
    bool headless = false;
};

struct CaptureStats {
    uint64_t frames = 0;
    uint64_t discarded_grabs = 0;
    uint64_t read_failures = 0;
    double elapsed_sec = 0.0;
    double fps_avg = 0.0;
    double fps_min = 0.0;
    double fps_max = 0.0;
    double frame_ms_avg = 0.0;
    double frame_ms_min = 0.0;
    double frame_ms_max = 0.0;
    int actual_width = 0;
    int actual_height = 0;
    bool buffer_size_set = false;
};

static std::string shell_escape_single_quote(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    return out;
}

static bool run_command_capture(const std::string& cmd, std::string& out) {
    out.clear();
    std::array<char, 512> buffer{};
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return false;
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        out += buffer.data();
    }
    const int rc = pclose(pipe);
    return rc == 0;
}

static std::string join_fps_list(const std::vector<double>& fps_list) {
    std::ostringstream oss;
    for (size_t i = 0; i < fps_list.size(); ++i) {
        if (i) oss << ',';
        const double v = fps_list[i];
        if (std::abs(v - static_cast<int>(v)) < 1e-9) oss << static_cast<int>(v);
        else oss << std::fixed << std::setprecision(2) << v;
    }
    return oss.str();
}

static bool probe_camera_modes(const std::string& device, ProbeResult& out, std::string& err) {
    out = {};
    out.device = device;

    const std::string dev_escaped = shell_escape_single_quote(device);

    std::string info;
    run_command_capture("v4l2-ctl -d '" + dev_escaped + "' --all 2>/dev/null", info);

    {
        std::regex card_re(R"(Card type\s*:\s*(.+))");
        std::regex bus_re(R"(Bus info\s*:\s*(.+))");
        std::smatch m;
        std::istringstream iss(info);
        std::string line;
        while (std::getline(iss, line)) {
            if (std::regex_search(line, m, card_re)) out.card_name = m[1].str();
            if (std::regex_search(line, m, bus_re)) out.bus_info = m[1].str();
        }
    }

    std::string formats;
    if (!run_command_capture("v4l2-ctl -d '" + dev_escaped + "' --list-formats-ext 2>/dev/null", formats)) {
        err = "failed to run v4l2-ctl --list-formats-ext";
        return false;
    }

    struct ModeKey {
        std::string pixel_format;
        int width = 0;
        int height = 0;
        bool operator<(const ModeKey& other) const {
            if (pixel_format != other.pixel_format) return pixel_format < other.pixel_format;
            if (width != other.width) return width < other.width;
            return height < other.height;
        }
    };

    std::map<ModeKey, std::set<double>> grouped;
    std::istringstream iss(formats);
    std::string line;
    std::string current_pixfmt;
    int current_width = 0;
    int current_height = 0;

    std::regex pix_re(R"(\[\d+\]:\s+'([^']+)')");
    std::regex size_re(R"(Size:\s+Discrete\s+(\d+)x(\d+))");
    std::regex fps_re(R"(([\d.]+)\s+fps)");
    std::smatch m;

    while (std::getline(iss, line)) {
        if (std::regex_search(line, m, pix_re)) {
            current_pixfmt = m[1].str();
            current_width = 0;
            current_height = 0;
            continue;
        }
        if (std::regex_search(line, m, size_re)) {
            current_width = std::stoi(m[1].str());
            current_height = std::stoi(m[2].str());
            grouped[{current_pixfmt, current_width, current_height}];
            continue;
        }
        if (std::regex_search(line, m, fps_re)) {
            if (!current_pixfmt.empty() && current_width > 0 && current_height > 0) {
                grouped[{current_pixfmt, current_width, current_height}].insert(std::stod(m[1].str()));
            }
        }
    }

    for (const auto& kv : grouped) {
        CameraMode mode;
        mode.pixel_format = kv.first.pixel_format;
        mode.width = kv.first.width;
        mode.height = kv.first.height;
        mode.fps_list.assign(kv.second.begin(), kv.second.end());
        out.modes.push_back(std::move(mode));
    }

    if (out.modes.empty()) {
        err = "no camera modes parsed from v4l2-ctl output";
        return false;
    }
    return true;
}

static void print_probe_result(const ProbeResult& probe) {
    std::cout << "\n=== Camera Probe Summary ===\n";
    std::cout << "Device      : " << probe.device << "\n";
    std::cout << "Card        : " << probe.card_name << "\n";
    std::cout << "Bus         : " << probe.bus_info << "\n";
    std::cout << "Modes       : " << probe.modes.size() << " unique format/resolution entries\n\n";

    std::cout << std::left
              << std::setw(8)  << "Format"
              << std::setw(12) << "Resolution"
              << std::setw(20) << "Supported FPS"
              << "\n";
    std::cout << std::string(40, '-') << "\n";

    for (const auto& m : probe.modes) {
        std::ostringstream res;
        res << m.width << 'x' << m.height;
        std::cout << std::setw(8)  << m.pixel_format
                  << std::setw(12) << res.str()
                  << std::setw(20) << join_fps_list(m.fps_list)
                  << "\n";
    }
    std::cout << "\n";
}

static int fourcc_from_string(const std::string& s) {
    if (s.size() != 4) return 0;
    return cv::VideoWriter::fourcc(s[0], s[1], s[2], s[3]);
}

class LatestFrameCamera {
public:
    bool open(const CameraConfig& cfg, std::string& err) {
        cfg_ = cfg;
        cap_.open(cfg.device, cv::CAP_V4L2);
        if (!cap_.isOpened()) {
            err = "cannot open camera: " + cfg.device;
            return false;
        }

        if (!cfg.fourcc.empty()) cap_.set(cv::CAP_PROP_FOURCC, fourcc_from_string(cfg.fourcc));
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);
        cap_.set(cv::CAP_PROP_FPS, cfg.fps);
        if (cfg.buffer_size > 0) {
            const bool ok = cap_.set(cv::CAP_PROP_BUFFERSIZE, cfg.buffer_size);
            stats_.buffer_size_set = ok;
        }

        for (int i = 0; i < std::max(0, cfg.warmup_frames); ++i) {
            cv::Mat tmp;
            cap_.read(tmp);
        }

        stats_.actual_width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
        stats_.actual_height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
        start_ = last_ = std::chrono::steady_clock::now();
        min_ms_ = 1e18;
        max_ms_ = 0.0;
        sum_ms_ = 0.0;
        return true;
    }

    bool read_latest(cv::Mat& frame, std::string& err) {
        if (!cap_.isOpened()) {
            err = "camera not open";
            return false;
        }

        if (cfg_.latest_only) {
            int successful_grabs = 0;
            const int drain = std::max(0, cfg_.drain_grabs);
            for (int i = 0; i < drain; ++i) {
                if (cap_.grab()) {
                    ++successful_grabs;
                } else {
                    break;
                }
            }
            if (successful_grabs > 1) stats_.discarded_grabs += static_cast<uint64_t>(successful_grabs - 1);
            if (!cap_.retrieve(frame)) {
                ++stats_.read_failures;
                err = "retrieve failed";
                return false;
            }
        } else {
            if (!cap_.read(frame)) {
                ++stats_.read_failures;
                err = "read failed";
                return false;
            }
        }

        if (frame.empty()) {
            ++stats_.read_failures;
            err = "empty frame";
            return false;
        }

        const auto now = std::chrono::steady_clock::now();
        const double dt_ms = std::chrono::duration<double, std::milli>(now - last_).count();
        last_ = now;

        if (stats_.frames > 0) {
            min_ms_ = std::min(min_ms_, dt_ms);
            max_ms_ = std::max(max_ms_, dt_ms);
            sum_ms_ += dt_ms;
        }

        ++stats_.frames;
        stats_.elapsed_sec = std::chrono::duration<double>(now - start_).count();
        if (stats_.elapsed_sec > 0.0) {
            stats_.fps_avg = static_cast<double>(stats_.frames) / stats_.elapsed_sec;
        }
        if (stats_.frames > 1) {
            stats_.frame_ms_avg = sum_ms_ / static_cast<double>(stats_.frames - 1);
            stats_.frame_ms_min = min_ms_;
            stats_.frame_ms_max = max_ms_;
            if (max_ms_ > 0.0) stats_.fps_min = 1000.0 / max_ms_;
            if (min_ms_ > 0.0 && min_ms_ < 1e17) stats_.fps_max = 1000.0 / min_ms_;
        }

        return true;
    }

    void close() {
        if (cap_.isOpened()) cap_.release();
    }

    const CaptureStats& stats() const { return stats_; }
    CaptureStats& stats() { return stats_; }
    bool is_open() const { return cap_.isOpened(); }

private:
    CameraConfig cfg_{};
    cv::VideoCapture cap_;
    CaptureStats stats_{};
    std::chrono::steady_clock::time_point start_{};
    std::chrono::steady_clock::time_point last_{};
    double min_ms_ = 1e18;
    double max_ms_ = 0.0;
    double sum_ms_ = 0.0;
};

} // namespace vision_app
