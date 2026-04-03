#include "camera.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace vision_app {
namespace {

#if defined(_WIN32)
constexpr const char* kDevNull = "NUL";
#define VISION_APP_POPEN _popen
#define VISION_APP_PCLOSE _pclose
#else
constexpr const char* kDevNull = "/dev/null";
#define VISION_APP_POPEN popen
#define VISION_APP_PCLOSE pclose
#endif

bool is_prefix(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

bool is_v4l2_device_path(const std::string& device) {
    return is_prefix(device, "/dev/video");
}

bool parse_int(const std::string& s, int& out) {
    if (s.empty()) return false;
    size_t pos = 0;
    try {
        int v = std::stoi(s, &pos);
        if (pos != s.size()) return false;
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

std::string device_to_v4l2_arg(const std::string& device) {
    if (is_v4l2_device_path(device)) return device;
    if (is_device_index_string(device)) return "/dev/video" + device;
    return device;
}

std::string shell_single_quote(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    out.push_back('\'');
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out.push_back(c);
    }
    out.push_back('\'');
    return out;
}

bool open_stream_probe(const std::string& device, CameraProbeResult& out, std::string& err) {
    cv::VideoCapture cap(device, cv::CAP_ANY);
    if (!cap.isOpened()) {
        err = "failed to open network stream";
        return false;
    }

    cv::Mat frame;
    for (int i = 0; i < 3; ++i) {
        if (cap.read(frame) && !frame.empty()) break;
    }

    CameraMode mode;
    mode.pixfmt = "stream";
    mode.width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    mode.height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    if ((mode.width <= 0 || mode.height <= 0) && !frame.empty()) {
        mode.width = frame.cols;
        mode.height = frame.rows;
    }
    const double fps = cap.get(cv::CAP_PROP_FPS);
    if (std::isfinite(fps) && fps > 0.0) mode.fps.push_back(fps);

    out.card = "OpenCV Probe Fallback";
    out.bus = "N/A";
    out.modes = {mode};
    err.clear();
    return true;
}

bool open_local_probe(const std::string& device, CameraProbeResult& out, std::string& err) {
    const std::string dev = device_to_v4l2_arg(device);

    std::string all;
    const std::string all_cmd = "v4l2-ctl -d " + shell_single_quote(dev) + " --all 2>" + kDevNull;
    run_command(all_cmd, all);

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
    const std::string fmt_cmd = "v4l2-ctl -d " + shell_single_quote(dev) + " --list-formats-ext 2>" + kDevNull;
    if (!run_command(fmt_cmd, formats)) {
        err = "failed to run v4l2-ctl --list-formats-ext";
        return false;
    }

    struct Key {
        std::string pixfmt;
        int w = 0;
        int h = 0;
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
    int cur_w = 0;
    int cur_h = 0;
    std::smatch m;
    std::istringstream iss(formats);
    std::string line;
    while (std::getline(iss, line)) {
        if (std::regex_search(line, m, pix_re)) {
            cur_pix = m[1].str();
            cur_w = 0;
            cur_h = 0;
            continue;
        }
        if (std::regex_search(line, m, size_re)) {
            cur_w = std::stoi(m[1].str());
            cur_h = std::stoi(m[2].str());
            grouped[{cur_pix, cur_w, cur_h}];
            continue;
        }
        if (std::regex_search(line, m, fps_re) && !cur_pix.empty() && cur_w > 0 && cur_h > 0) {
            grouped[{cur_pix, cur_w, cur_h}].insert(std::stod(m[1].str()));
        }
    }

    out.modes.clear();
    for (const auto& kv : grouped) {
        CameraMode mode;
        mode.pixfmt = kv.first.pixfmt;
        mode.width = kv.first.w;
        mode.height = kv.first.h;
        mode.fps.assign(kv.second.begin(), kv.second.end());
        out.modes.push_back(mode);
    }

    if (out.modes.empty()) {
        err = "no camera modes parsed";
        return false;
    }

    err.clear();
    return true;
}

bool open_local_capture(cv::VideoCapture& cap, const std::string& device) {
    int index = -1;
    if (parse_int(device, index)) {
        if (cap.open(index, cv::CAP_V4L2)) return true;
        cap.release();
        if (cap.open(index, cv::CAP_ANY)) return true;
        cap.release();
        return false;
    }

    if (cap.open(device, cv::CAP_V4L2)) return true;
    cap.release();
    if (cap.open(device, cv::CAP_ANY)) return true;
    cap.release();
    return false;
}

void try_prime_capture_size(cv::VideoCapture& cap, int& width, int& height) {
    width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    if (width > 0 && height > 0) return;

    cv::Mat frame;
    if (cap.read(frame) && !frame.empty()) {
        width = frame.cols;
        height = frame.rows;
    }
}

} // namespace

bool is_stream_url(const std::string& device) {
    return is_prefix(device, "rtsp://") ||
           is_prefix(device, "http://") ||
           is_prefix(device, "https://");
}

bool is_device_index_string(const std::string& device) {
    int dummy = -1;
    return parse_int(device, dummy);
}

bool run_command(const std::string& cmd, std::string& out) {
    out.clear();
    std::array<char, 512> buf{};
    FILE* fp = VISION_APP_POPEN(cmd.c_str(), "r");
    if (!fp) return false;
    while (std::fgets(buf.data(), static_cast<int>(buf.size()), fp)) out += buf.data();
    const int rc = VISION_APP_PCLOSE(fp);
    return rc == 0;
}

bool probe_camera(const std::string& device, CameraProbeResult& out, std::string& err) {
    out = {};
    out.device = device;

    if (is_stream_url(device)) {
        std::cout << "Device is not /dev/video*, falling back to OpenCV probe...\n";
        return open_stream_probe(device, out, err);
    }

    return open_local_probe(device, out, err);
}

void print_probe(const CameraProbeResult& p) {
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
        for (size_t i = 0; i < m.fps.size(); ++i) {
            if (i) fps << ',';
            const double v = m.fps[i];
            if (std::abs(v - std::round(v)) < 1e-6) fps << static_cast<int>(std::round(v));
            else fps << v;
        }
        std::cout << std::setw(8) << m.pixfmt << std::setw(12) << res.str() << fps.str() << '\n';
    }
}

int fourcc_from_string(const std::string& s) {
    if (s.size() != 4) return 0;
    return cv::VideoWriter::fourcc(s[0], s[1], s[2], s[3]);
}

void clamp_camera_size(int& w, int& h, int soft_max) {
    const int m = std::max(w, h);
    if (m <= soft_max || soft_max <= 0) return;
    const double scale = static_cast<double>(soft_max) / static_cast<double>(m);
    w = std::max(1, static_cast<int>(std::round(w * scale)));
    h = std::max(1, static_cast<int>(std::round(h * scale)));
}

bool open_capture(cv::VideoCapture& cap,
                  const std::string& device,
                  int& width,
                  int& height,
                  int fps,
                  const std::string& fourcc,
                  int buffer_size,
                  int camera_soft_max,
                  std::string& err) {
    clamp_camera_size(width, height, camera_soft_max);

    const bool url = is_stream_url(device);
    if (url) {
        cap.open(device, cv::CAP_ANY);
    } else {
        if (!open_local_capture(cap, device)) {
            err = "cannot open camera: " + device;
            return false;
        }
    }

    if (!cap.isOpened()) {
        err = "cannot open camera: " + device;
        return false;
    }

    if (!url) {
        if (!fourcc.empty()) cap.set(cv::CAP_PROP_FOURCC, fourcc_from_string(fourcc));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        if (fps > 0) cap.set(cv::CAP_PROP_FPS, fps);
        if (buffer_size > 0) cap.set(cv::CAP_PROP_BUFFERSIZE, buffer_size);
    }

    try_prime_capture_size(cap, width, height);
    err.clear();
    return true;
}

bool grab_latest_frame(cv::VideoCapture& cap, bool latest_only, int drain_grabs, cv::Mat& frame) {
    if (!latest_only) return cap.read(frame) && !frame.empty();
    if (!cap.grab()) return false;
    for (int i = 0; i < drain_grabs; ++i) {
        if (!cap.grab()) break;
    }
    return cap.retrieve(frame) && !frame.empty();
}

cv::Mat downscale_for_preview(const cv::Mat& src, int preview_soft_max) {
    if (src.empty()) return src;
    const int m = std::max(src.cols, src.rows);
    if (m <= preview_soft_max || preview_soft_max <= 0) return src;
    const double scale = static_cast<double>(preview_soft_max) / static_cast<double>(m);
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(), scale, scale, cv::INTER_AREA);
    return dst;
}

bool bench_capture(const std::string& device,
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
    for (int i = 0; i < 5; ++i) grab_latest_frame(cap, latest_only, drain_grabs, frame);

    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    auto last = t0;
    double dt_min = 1e30, dt_max = 0.0, dt_sum = 0.0;

    if (!headless) cv::namedWindow("vision_app", cv::WINDOW_AUTOSIZE);

    while (true) {
        if (!grab_latest_frame(cap, latest_only, drain_grabs, frame)) {
            err = "failed to read frame";
            return false;
        }

        const auto now = clk::now();
        const double dt = std::chrono::duration<double, std::milli>(now - last).count();
        last = now;
        if (stats.frames > 0) {
            dt_min = std::min(dt_min, dt);
            dt_max = std::max(dt_max, dt);
            dt_sum += dt;
        }
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
    err.clear();
    return true;
}

} // namespace vision_app
