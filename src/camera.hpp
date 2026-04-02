#pragma once

#include <algorithm>
#include <array>
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
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace vision_app {

struct CameraMode {
    std::string pixfmt;
    int width = 0;
    int height = 0;
    std::vector<double> fps;
};

struct CameraProbeResult {
    std::string device;
    std::string card;
    std::string bus;
    std::vector<CameraMode> modes;
};

struct RuntimeStats {
    uint64_t frames = 0;
    double elapsed_sec = 0.0;
    double fps_avg = 0.0;
    double fps_min = 0.0;
    double fps_max = 0.0;
    double frame_time_avg_ms = 0.0;
    double frame_time_min_ms = 0.0;
    double frame_time_max_ms = 0.0;
    int actual_width = 0;
    int actual_height = 0;
};

inline bool run_command(const std::string& cmd, std::string& out) {
    out.clear();
    std::array<char, 512> buf{};
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) return false;
    while (fgets(buf.data(), static_cast<int>(buf.size()), fp)) out += buf.data();
    const int rc = pclose(fp);
    return rc == 0;
}

inline bool probe_camera(const std::string& device, CameraProbeResult& out, std::string& err) {
    out = {};
    out.device = device;

    bool is_v4l2 = (device.find("/dev/video") == 0);
    if (!is_v4l2) {
        std::cout << "Device is not /dev/video*, falling back to OpenCV probe...\n";
        cv::VideoCapture cap;
        bool is_number = !device.empty() && std::all_of(device.begin(), device.end(), ::isdigit);
        if (is_number) cap.open(std::stoi(device), cv::CAP_ANY);
        else cap.open(device, cv::CAP_ANY);
        
        if (!cap.isOpened()) {
            err = "cannot open non-v4l2 camera: " + device;
            return false;
        }
        
        CameraMode mode;
        int fcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
        char fcc_str[5] = {0};
        for (int i = 0; i < 4; ++i) {
            fcc_str[i] = static_cast<char>((fcc >> (i * 8)) & 0xFF);
            if (!isprint(fcc_str[i])) fcc_str[i] = ' ';
        }
        if (std::string(fcc_str) == "    ") mode.pixfmt = "UKNW";
        else mode.pixfmt = fcc_str;
        mode.width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        mode.height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps > 0) mode.fps.push_back(fps);
        else mode.fps.push_back(0.0);
        
        out.card = "OpenCV Probe Fallback";
        out.bus = "N/A";
        out.modes.push_back(mode);
        return true;
    }

    std::string all;
    run_command("v4l2-ctl -d '" + device + "' --all 2>/dev/null", all);
    {
        std::regex card_re(R"(Card type\s*:\s*(.+))");
        std::regex bus_re(R"(Bus info\s*:\s*(.+))");
        std::smatch m;
        std::istringstream iss(all);
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
        std::string pixfmt; int w=0; int h=0;
        bool operator<(const Key& o) const {
            if (pixfmt != o.pixfmt) return pixfmt < o.pixfmt;
            if (w != o.w) return w < o.w;
            return h < o.h;
        }
    };
    std::map<Key, std::set<double>> grouped;
    std::regex pix_re(R"(\[\d+\]:\s+'([^']+)')");
    std::regex size_re(R"(Size:\s+Discrete\s+(\d+)x(\d+))");
    std::regex fps_re(R"(([\d.]+)\s+fps)");
    std::string cur_pix;
    int cur_w = 0, cur_h = 0;
    std::smatch m;
    std::istringstream iss(formats);
    std::string line;
    while (std::getline(iss, line)) {
        if (std::regex_search(line, m, pix_re)) { cur_pix = m[1].str(); cur_w = cur_h = 0; continue; }
        if (std::regex_search(line, m, size_re)) { cur_w = std::stoi(m[1].str()); cur_h = std::stoi(m[2].str()); grouped[{cur_pix, cur_w, cur_h}]; continue; }
        if (std::regex_search(line, m, fps_re) && !cur_pix.empty() && cur_w > 0 && cur_h > 0) grouped[{cur_pix, cur_w, cur_h}].insert(std::stod(m[1].str()));
    }
    for (const auto& kv : grouped) {
        CameraMode mode; mode.pixfmt = kv.first.pixfmt; mode.width = kv.first.w; mode.height = kv.first.h; mode.fps.assign(kv.second.begin(), kv.second.end()); out.modes.push_back(mode);
    }
    if (out.modes.empty()) { err = "no camera modes parsed"; return false; }
    return true;
}

inline void print_probe(const CameraProbeResult& p) {
    std::cout << "=== Camera Probe Summary ===\n";
    std::cout << "Device : " << p.device << "\n";
    std::cout << "Card   : " << p.card << "\n";
    std::cout << "Bus    : " << p.bus << "\n";
    std::cout << "Modes  : " << p.modes.size() << " unique entries\n\n";
    std::cout << std::left << std::setw(8) << "Format" << std::setw(12) << "Resolution" << "FPS\n";
    std::cout << "--------------------------------------------\n";
    for (const auto& m : p.modes) {
        std::ostringstream res, fps;
        res << m.width << 'x' << m.height;
        for (size_t i=0;i<m.fps.size();++i) {
            if (i) fps << ',';
            const double v = m.fps[i];
            if (std::abs(v - std::round(v)) < 1e-6) fps << static_cast<int>(std::round(v));
            else fps << v;
        }
        std::cout << std::setw(8) << m.pixfmt << std::setw(12) << res.str() << fps.str() << '\n';
    }
}

inline int fourcc_from_string(const std::string& s) {
    if (s.size() != 4) return 0;
    return cv::VideoWriter::fourcc(s[0], s[1], s[2], s[3]);
}

inline void clamp_camera_size(int& w, int& h, int soft_max) {
    const int m = std::max(w, h);
    if (m <= soft_max || soft_max <= 0) return;
    const double scale = static_cast<double>(soft_max) / static_cast<double>(m);
    w = std::max(1, static_cast<int>(std::round(w * scale)));
    h = std::max(1, static_cast<int>(std::round(h * scale)));
}

inline bool open_capture(cv::VideoCapture& cap,
                         const std::string& device,
                         int& width,
                         int& height,
                         int fps,
                         const std::string& fourcc,
                         int buffer_size,
                         int camera_soft_max,
                         std::string& err) {
    clamp_camera_size(width, height, camera_soft_max);
    
    bool is_number = !device.empty() && std::all_of(device.begin(), device.end(), ::isdigit);
    int backend = cv::CAP_ANY;
    if (device.find("/dev/video") == 0) {
        backend = cv::CAP_V4L2;
    }
    
    if (is_number) {
        cap.open(std::stoi(device), backend);
    } else {
        cap.open(device, backend);
    }
    
    if (!cap.isOpened()) { err = "cannot open camera: " + device; return false; }
    if (!fourcc.empty()) cap.set(cv::CAP_PROP_FOURCC, fourcc_from_string(fourcc));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    if (fps > 0) cap.set(cv::CAP_PROP_FPS, fps);
    if (buffer_size > 0) cap.set(cv::CAP_PROP_BUFFERSIZE, buffer_size);
    width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    return true;
}

inline bool grab_latest_frame(cv::VideoCapture& cap, bool latest_only, int drain_grabs, cv::Mat& frame) {
    if (!latest_only) return cap.read(frame) && !frame.empty();
    if (!cap.grab()) return false;
    for (int i = 0; i < drain_grabs; ++i) {
        if (!cap.grab()) break;
    }
    return cap.retrieve(frame) && !frame.empty();
}

inline cv::Mat downscale_for_preview(const cv::Mat& src, int preview_soft_max) {
    if (src.empty()) return src;
    const int m = std::max(src.cols, src.rows);
    if (m <= preview_soft_max || preview_soft_max <= 0) return src;
    const double scale = static_cast<double>(preview_soft_max) / static_cast<double>(m);
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(), scale, scale, cv::INTER_AREA);
    return dst;
}

inline bool bench_capture(const std::string& device,
                          int width,
                          int height,
                          int fps,
                          const std::string& fourcc,
                          int buffer_size,
                          bool latest_only,
                          int drain_grabs,
                          bool headless,
                          int duration_sec,
                          int camera_soft_max,
                          int preview_soft_max,
                          RuntimeStats& stats,
                          std::string& err) {
    stats = {};
    cv::VideoCapture cap;
    if (!open_capture(cap, device, width, height, fps, fourcc, buffer_size, camera_soft_max, err)) return false;
    cv::Mat frame;
    for (int i=0;i<5;++i) grab_latest_frame(cap, latest_only, drain_grabs, frame);
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    auto last = t0;
    double dt_min = 1e30, dt_max = 0.0, dt_sum = 0.0;
    if (!headless) cv::namedWindow("vision_app", cv::WINDOW_AUTOSIZE);
    while (true) {
        if (!grab_latest_frame(cap, latest_only, drain_grabs, frame)) { err = "failed to read frame"; return false; }
        const auto now = clk::now();
        const double dt = std::chrono::duration<double, std::milli>(now - last).count();
        last = now;
        if (stats.frames > 0) { dt_min = std::min(dt_min, dt); dt_max = std::max(dt_max, dt); dt_sum += dt; }
        ++stats.frames;
        if (!headless) {
            cv::Mat show = downscale_for_preview(frame, preview_soft_max);
            cv::imshow("vision_app", show);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) break;
        }
        if (std::chrono::duration<double>(now - t0).count() >= duration_sec) break;
    }
    stats.elapsed_sec = std::chrono::duration<double>(clk::now() - t0).count();
    stats.actual_width = frame.cols;
    stats.actual_height = frame.rows;
    if (stats.elapsed_sec > 0.0) stats.fps_avg = static_cast<double>(stats.frames) / stats.elapsed_sec;
    if (stats.frames > 1) {
        stats.frame_time_avg_ms = dt_sum / static_cast<double>(stats.frames - 1);
        stats.frame_time_min_ms = dt_min;
        stats.frame_time_max_ms = dt_max;
        if (dt_max > 0.0) stats.fps_min = 1000.0 / dt_max;
        if (dt_min > 0.0) stats.fps_max = 1000.0 / dt_min;
    }
    if (!headless) cv::destroyAllWindows();
    return true;
}

} // namespace vision_app
