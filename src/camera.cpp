#include "camera.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace app {

CameraDevice::~CameraDevice() { close(); }

void ensureParentDir(const std::string& path) {
    std::filesystem::path p(path);
    if (!p.parent_path().empty()) {
        std::filesystem::create_directories(p.parent_path());
    }
}

void drawStatusText(cv::Mat& frame, const std::string& text, int line, const cv::Scalar& color) {
    const int y = 24 + line * 24;
    cv::putText(frame, text, cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, 0.65, {0,0,0}, 3, cv::LINE_AA);
    cv::putText(frame, text, cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv::LINE_AA);
}

int CameraDevice::fourccToInt(const std::string& fourcc) {
    if (fourcc.size() != 4) return 0;
    return cv::VideoWriter::fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3]);
}

std::string CameraDevice::intToFourcc(int v) {
    char c[] = {
        static_cast<char>(v & 255),
        static_cast<char>((v >> 8) & 255),
        static_cast<char>((v >> 16) & 255),
        static_cast<char>((v >> 24) & 255),
        0
    };
    return std::string(c);
}

bool CameraDevice::applyRequestedMode(const CameraConfig& cfg) {
    cap_.set(cv::CAP_PROP_BUFFERSIZE, cfg.buffer_size);
    if (!cfg.fourcc.empty()) {
        cap_.set(cv::CAP_PROP_FOURCC, static_cast<double>(fourccToInt(cfg.fourcc)));
    }
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);
    cap_.set(cv::CAP_PROP_FPS, cfg.fps);
    return true;
}

CameraConfig CameraDevice::readBackActualConfig() const {
    CameraConfig a = requested_;
    a.actual_width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    a.actual_height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    a.actual_fps = cap_.get(cv::CAP_PROP_FPS);
    a.actual_fourcc = intToFourcc(static_cast<int>(cap_.get(cv::CAP_PROP_FOURCC)));
    return a;
}

bool CameraDevice::open(const CameraConfig& cfg, std::string* err) {
    close();
    requested_ = cfg;
    if (!cap_.open(cfg.device_index, cfg.backend)) {
        if (err) *err = "cannot open camera";
        return false;
    }
    applyRequestedMode(cfg);
    actual_ = readBackActualConfig();
    open_ = true;
    return true;
}

void CameraDevice::close() {
    if (cap_.isOpened()) cap_.release();
    open_ = false;
}

bool CameraDevice::isOpen() const { return open_ && cap_.isOpened(); }

void CameraDevice::applyConfiguredFlips(cv::Mat& frame) const {
    if (requested_.flip_horizontal && requested_.flip_vertical) cv::flip(frame, frame, -1);
    else if (requested_.flip_horizontal) cv::flip(frame, frame, 1);
    else if (requested_.flip_vertical) cv::flip(frame, frame, 0);
}

bool CameraDevice::read(cv::Mat& frame, std::string* err) {
    if (!isOpen()) {
        if (err) *err = "camera not open";
        return false;
    }
    if (!cap_.read(frame) || frame.empty()) {
        if (err) *err = "camera read failed";
        return false;
    }
    applyConfiguredFlips(frame);
    return true;
}

double CameraDevice::measureFps(int warmup_frames, int measure_frames, std::string* err) {
    cv::Mat frame;
    for (int i = 0; i < warmup_frames; ++i) {
        if (!read(frame, err)) return 0.0;
    }
    const int64_t t0 = cv::getTickCount();
    int ok = 0;
    for (int i = 0; i < measure_frames; ++i) {
        if (!read(frame, err)) break;
        ++ok;
    }
    const double dt = (cv::getTickCount() - t0) / cv::getTickFrequency();
    if (dt <= 0.0) return 0.0;
    return ok / dt;
}

CameraConfig CameraDevice::actualConfig() const { return actual_; }

static std::string shellEscape(const std::string& s) {
    std::string out = "'";
    for (char ch : s) {
        if (ch == '\'') out += "'\\''";
        else out.push_back(ch);
    }
    out += "'";
    return out;
}

bool ProbeRunner::writeV4L2Report(const AppConfig& cfg, std::string* err) {
    ensureParentDir(cfg.camera.probe_report_path);
    std::ostringstream cmd;
    cmd << "bash -lc \"v4l2-ctl -d /dev/video" << cfg.camera.device_index
        << " --list-formats-ext > " << shellEscape(cfg.camera.probe_report_path) << "\"";
    const int rc = std::system(cmd.str().c_str());
    if (rc != 0 && err) *err = "v4l2-ctl command failed";
    return rc == 0;
}

bool ProbeRunner::runOpenCvProbe(const AppConfig& cfg, std::vector<ProbeRow>& rows, std::string* err) {
    const std::vector<CandidateMode> candidates = {
        {320,240,30,"MJPG"}, {640,480,30,"MJPG"}, {640,480,60,"MJPG"},
        {1280,720,30,"MJPG"}, {1920,1080,30,"MJPG"}
    };

    rows.clear();
    for (const auto& c : candidates) {
        ProbeRow row{};
        row.requested_width = c.width;
        row.requested_height = c.height;
        row.requested_fps = c.fps;
        row.requested_fourcc = c.fourcc;

        CameraConfig cam_cfg = cfg.camera;
        cam_cfg.width = c.width;
        cam_cfg.height = c.height;
        cam_cfg.fps = c.fps;
        cam_cfg.fourcc = c.fourcc;

        CameraDevice cam;
        std::string local_err;
        row.open_ok = cam.open(cam_cfg, &local_err);
        if (!row.open_ok) {
            row.notes = local_err;
            rows.push_back(row);
            continue;
        }
        const auto actual = cam.actualConfig();
        row.actual_width = actual.actual_width;
        row.actual_height = actual.actual_height;
        row.actual_fps = actual.actual_fps;
        row.actual_fourcc = actual.actual_fourcc;
        cv::Mat frame;
        row.read_ok = cam.read(frame, &local_err);
        if (row.read_ok) row.measured_loop_fps = cam.measureFps(30, 120, &local_err);
        row.notes = local_err;
        rows.push_back(row);
    }
    return true;
}

bool ProbeRunner::writeCsv(const std::string& path, const std::vector<ProbeRow>& rows, std::string* err) {
    try {
        ensureParentDir(path);
        std::ofstream ofs(path);
        ofs << "req_w,req_h,req_fps,req_fourcc,act_w,act_h,act_fps,act_fourcc,measured_loop_fps,open_ok,read_ok,notes\n";
        for (const auto& r : rows) {
            ofs << r.requested_width << ',' << r.requested_height << ',' << r.requested_fps << ',' << r.requested_fourcc << ','
                << r.actual_width << ',' << r.actual_height << ',' << std::fixed << std::setprecision(2) << r.actual_fps << ','
                << r.actual_fourcc << ',' << r.measured_loop_fps << ',' << r.open_ok << ',' << r.read_ok << ',' << '"' << r.notes << '"' << "\n";
        }
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

bool ProbeRunner::writeJson(const std::string& path, const AppConfig& cfg, const std::vector<ProbeRow>& rows, std::string* err) {
    try {
        ensureParentDir(path);
        nlohmann::json j;
        j["device_index"] = cfg.camera.device_index;
        j["rows"] = nlohmann::json::array();
        for (const auto& r : rows) {
            j["rows"].push_back({
                {"requested", {{"width", r.requested_width}, {"height", r.requested_height}, {"fps", r.requested_fps}, {"fourcc", r.requested_fourcc}}},
                {"actual", {{"width", r.actual_width}, {"height", r.actual_height}, {"fps", r.actual_fps}, {"fourcc", r.actual_fourcc}}},
                {"measured_loop_fps", r.measured_loop_fps},
                {"open_ok", r.open_ok},
                {"read_ok", r.read_ok},
                {"notes", r.notes}
            });
        }
        std::ofstream ofs(path);
        ofs << j.dump(2);
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

} // namespace app
