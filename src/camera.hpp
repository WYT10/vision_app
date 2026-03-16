#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
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
    std::string pixfmt;
    int width = 0;
    int height = 0;
    std::vector<double> fps_list;
};

struct CameraProbeResult {
    std::string device;
    std::string card;
    std::string bus;
    std::vector<CameraMode> modes;
};

struct CameraOptions {
    std::string device = "/dev/video0";
    int width = 1280;
    int height = 720;
    int fps = 30;
    std::string fourcc = "MJPG";
    int buffer_size = 1;
    bool latest_only = true;
    int drain_grabs = 2;
    bool headless = false;
};

static inline bool run_command(const std::string& cmd, std::string& out) {
    out.clear();
    std::array<char, 512> buf{};
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) return false;
    while (fgets(buf.data(), static_cast<int>(buf.size()), fp)) out += buf.data();
    return pclose(fp) == 0;
}

static inline int fourcc_from_string(const std::string& s) {
    if (s.size() != 4) return 0;
    return cv::VideoWriter::fourcc(s[0], s[1], s[2], s[3]);
}

static inline std::vector<double> unique_sorted(const std::vector<double>& xs) {
    std::set<double> s(xs.begin(), xs.end());
    return {s.begin(), s.end()};
}

static inline bool probe_camera(const std::string& device, CameraProbeResult& out, std::string& err) {
    out = {};
    out.device = device;
    std::string info;
    run_command("v4l2-ctl -d '" + device + "' --all 2>/dev/null", info);
    {
        std::regex card_re(R"(Card type\s*:\s*(.+))");
        std::regex bus_re(R"(Bus info\s*:\s*(.+))");
        std::smatch m;
        std::istringstream iss(info);
        std::string line;
        while (std::getline(iss, line)) {
            if (std::regex_search(line, m, card_re)) out.card = m[1].str();
            if (std::regex_search(line, m, bus_re)) out.bus = m[1].str();
        }
    }

    std::string formats;
    if (!run_command("v4l2-ctl -d '" + device + "' --list-formats-ext 2>/dev/null", formats)) {
        err = "failed to run v4l2-ctl --list-formats-ext";
        return false;
    }

    struct Key {
        std::string fmt;
        int w = 0;
        int h = 0;
        bool operator<(const Key& o) const {
            if (fmt != o.fmt) return fmt < o.fmt;
            if (w != o.w) return w < o.w;
            return h < o.h;
        }
    };
    std::map<Key, std::set<double>> grouped;

    std::regex pix_re(R"(\[\d+\]:\s+'([^']+)')");
    std::regex size_re(R"(Size:\s+Discrete\s+(\d+)x(\d+))");
    std::regex fps_re(R"(([\d.]+)\s+fps)");
    std::smatch m;
    std::istringstream iss(formats);
    std::string line;
    std::string cur_fmt;
    int cur_w = 0, cur_h = 0;
    while (std::getline(iss, line)) {
        if (std::regex_search(line, m, pix_re)) {
            cur_fmt = m[1].str();
            cur_w = cur_h = 0;
        } else if (std::regex_search(line, m, size_re)) {
            cur_w = std::stoi(m[1].str());
            cur_h = std::stoi(m[2].str());
            grouped[{cur_fmt, cur_w, cur_h}];
        } else if (std::regex_search(line, m, fps_re)) {
            if (!cur_fmt.empty() && cur_w > 0 && cur_h > 0) {
                grouped[{cur_fmt, cur_w, cur_h}].insert(std::stod(m[1].str()));
            }
        }
    }

    for (const auto& kv : grouped) {
        CameraMode mode;
        mode.pixfmt = kv.first.fmt;
        mode.width = kv.first.w;
        mode.height = kv.first.h;
        mode.fps_list.assign(kv.second.begin(), kv.second.end());
        out.modes.push_back(mode);
    }
    if (out.modes.empty()) {
        err = "no camera modes parsed";
        return false;
    }
    return true;
}

static inline void print_probe(const CameraProbeResult& pr) {
    std::cout << "\n=== Camera Probe Summary ===\n";
    std::cout << "Device : " << pr.device << "\n";
    std::cout << "Card   : " << pr.card << "\n";
    std::cout << "Bus    : " << pr.bus << "\n";
    std::cout << "Modes  : " << pr.modes.size() << " unique entries\n\n";
    std::cout << std::left << std::setw(8) << "Format" << std::setw(12) << "Resolution" << "FPS\n";
    std::cout << std::string(44, '-') << "\n";
    for (const auto& m : pr.modes) {
        std::ostringstream fps;
        auto vals = unique_sorted(m.fps_list);
        for (size_t i = 0; i < vals.size(); ++i) {
            if (i) fps << ',';
            if (std::abs(vals[i] - std::round(vals[i])) < 1e-9) fps << static_cast<int>(std::round(vals[i]));
            else fps << vals[i];
        }
        std::ostringstream res;
        res << m.width << 'x' << m.height;
        std::cout << std::setw(8) << m.pixfmt << std::setw(12) << res.str() << fps.str() << '\n';
    }
    std::cout << '\n';
}

class CameraCapture {
public:
    bool open(const CameraOptions& opt, std::string& err) {
        opt_ = opt;
        cap_.open(opt.device, cv::CAP_V4L2);
        if (!cap_.isOpened()) {
            err = "cannot open camera: " + opt.device;
            return false;
        }
        if (!opt.fourcc.empty()) cap_.set(cv::CAP_PROP_FOURCC, fourcc_from_string(opt.fourcc));
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, opt.width);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, opt.height);
        cap_.set(cv::CAP_PROP_FPS, opt.fps);
        cap_.set(cv::CAP_PROP_BUFFERSIZE, std::max(1, opt.buffer_size));
        return true;
    }

    bool read_latest(cv::Mat& frame) {
        if (!cap_.isOpened()) return false;
        if (opt_.latest_only) {
            int successful = 0;
            for (int i = 0; i < std::max(0, opt_.drain_grabs); ++i) {
                if (cap_.grab()) ++successful;
            }
            (void)successful;
            if (!cap_.retrieve(frame)) return false;
            return !frame.empty();
        }
        return cap_.read(frame) && !frame.empty();
    }

    void release() { cap_.release(); }
    bool is_open() const { return cap_.isOpened(); }

private:
    CameraOptions opt_{};
    cv::VideoCapture cap_{};
};

} // namespace vision_app
