#pragma once

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "camera.hpp"
#include "calibrate.hpp"

namespace vision_app {

struct StageStats {
    uint64_t frames = 0;
    uint64_t trigger_count = 0;
    double detect_ms_avg = 0.0;
    double warp_ms_avg = 0.0;
    double infer_ms_avg = 0.0;
    double total_ms_avg = 0.0;
    double detect_ms_max = 0.0;
    double warp_ms_max = 0.0;
    double infer_ms_max = 0.0;
    double total_ms_max = 0.0;
    double detector_fps_avg = 0.0;
    double last_red_mean = 0.0;
    double last_red_score = 0.0;
};

struct RollingTimer {
    uint64_t n = 0;
    double sum_ms = 0.0;
    double max_ms = 0.0;
    void push(double v) {
        ++n;
        sum_ms += v;
        if (v > max_ms) max_ms = v;
    }
    double avg() const { return n ? sum_ms / static_cast<double>(n) : 0.0; }
};

static std::string now_iso8601_basic() {
    using clock = std::chrono::system_clock;
    const auto now = clock::now();
    const std::time_t t = clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

static void print_capture_stats(const CameraConfig& cfg, const CaptureStats& s) {
    std::cout << "\n=== Runtime stats ===\n";
    std::cout << "Device            : " << cfg.device << "\n";
    std::cout << "Requested mode    : " << cfg.width << "x" << cfg.height << " @ " << cfg.fps << " fps, " << cfg.fourcc << "\n";
    std::cout << "Actual frame size : " << s.actual_width << "x" << s.actual_height << "\n";
    std::cout << "Frames            : " << s.frames << "\n";
    std::cout << "Elapsed sec       : " << s.elapsed_sec << "\n";
    std::cout << "FPS avg           : " << s.fps_avg << "\n";
    std::cout << "FPS min           : " << s.fps_min << "\n";
    std::cout << "FPS max           : " << s.fps_max << "\n";
    std::cout << "Frame ms avg      : " << s.frame_ms_avg << "\n";
    std::cout << "Frame ms min      : " << s.frame_ms_min << "\n";
    std::cout << "Frame ms max      : " << s.frame_ms_max << "\n";
    std::cout << "Discarded grabs   : " << s.discarded_grabs << "\n";
    std::cout << "Read failures     : " << s.read_failures << "\n";
    std::cout << "Buffer size set   : " << (s.buffer_size_set ? "yes" : "no") << "\n";
}

static bool append_test_csv(const std::string& path,
                            const CameraConfig& cfg,
                            const CaptureStats& s,
                            const StageStats* stage = nullptr,
                            const HomographyLock* lock = nullptr) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    const bool exists = std::filesystem::exists(path);
    std::ofstream out(path, std::ios::app);
    if (!out.is_open()) return false;

    if (!exists) {
        out << "timestamp,device,width_req,height_req,fps_req,fourcc,width_actual,height_actual,frames,elapsed_sec,fps_avg,fps_min,fps_max,frame_ms_avg,frame_ms_min,frame_ms_max,discarded_grabs,read_failures,detect_ms_avg,warp_ms_avg,infer_ms_avg,total_ms_avg,detect_ms_max,warp_ms_max,infer_ms_max,total_ms_max,detector_fps_avg,trigger_count,last_red_mean,last_red_score,tag_id,warp_width,warp_height\n";
    }

    out << now_iso8601_basic() << ','
        << cfg.device << ','
        << cfg.width << ','
        << cfg.height << ','
        << cfg.fps << ','
        << cfg.fourcc << ','
        << s.actual_width << ','
        << s.actual_height << ','
        << s.frames << ','
        << s.elapsed_sec << ','
        << s.fps_avg << ','
        << s.fps_min << ','
        << s.fps_max << ','
        << s.frame_ms_avg << ','
        << s.frame_ms_min << ','
        << s.frame_ms_max << ','
        << s.discarded_grabs << ','
        << s.read_failures << ','
        << (stage ? stage->detect_ms_avg : 0.0) << ','
        << (stage ? stage->warp_ms_avg : 0.0) << ','
        << (stage ? stage->infer_ms_avg : 0.0) << ','
        << (stage ? stage->total_ms_avg : 0.0) << ','
        << (stage ? stage->detect_ms_max : 0.0) << ','
        << (stage ? stage->warp_ms_max : 0.0) << ','
        << (stage ? stage->infer_ms_max : 0.0) << ','
        << (stage ? stage->total_ms_max : 0.0) << ','
        << (stage ? stage->detector_fps_avg : 0.0) << ','
        << (stage ? stage->trigger_count : 0) << ','
        << (stage ? stage->last_red_mean : 0.0) << ','
        << (stage ? stage->last_red_score : 0.0) << ','
        << (lock && lock->valid ? lock->id : -1) << ','
        << (lock && lock->valid ? lock->warp_size.width : 0) << ','
        << (lock && lock->valid ? lock->warp_size.height : 0)
        << '\n';

    return true;
}

static bool write_probe_csv(const std::string& path, const ProbeResult& probe) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path, std::ios::trunc);
    if (!out.is_open()) return false;
    out << "device,card,bus,pixel_format,width,height,supported_fps\n";
    for (const auto& m : probe.modes) {
        out << probe.device << ','
            << '"' << probe.card_name << '"' << ','
            << '"' << probe.bus_info << '"' << ','
            << m.pixel_format << ','
            << m.width << ','
            << m.height << ','
            << '"' << join_fps_list(m.fps_list) << '"' << '\n';
    }
    return true;
}

static bool write_latest_report_md(const std::string& path,
                                   const CameraConfig& cfg,
                                   const CaptureStats& capture,
                                   const StageStats* stage,
                                   const HomographyLock* lock,
                                   const RoiSet* rois) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path, std::ios::trunc);
    if (!out.is_open()) return false;

    out << "# Vision App Report\n\n";
    out << "## Camera\n";
    out << "- Device: `" << cfg.device << "`\n";
    out << "- Requested: `" << cfg.width << 'x' << cfg.height << " @ " << cfg.fps << " " << cfg.fourcc << "`\n";
    out << "- Actual: `" << capture.actual_width << 'x' << capture.actual_height << "`\n\n";

    out << "## Capture Stats\n";
    out << "- Frames: " << capture.frames << "\n";
    out << "- Elapsed sec: " << capture.elapsed_sec << "\n";
    out << "- FPS avg: " << capture.fps_avg << "\n";
    out << "- FPS min/max: " << capture.fps_min << " / " << capture.fps_max << "\n";
    out << "- Frame ms avg/min/max: " << capture.frame_ms_avg << " / " << capture.frame_ms_min << " / " << capture.frame_ms_max << "\n";
    out << "- Discarded grabs: " << capture.discarded_grabs << "\n\n";

    if (stage) {
        out << "## Processing Stats\n";
        out << "- Detect ms avg/max: " << stage->detect_ms_avg << " / " << stage->detect_ms_max << "\n";
        out << "- Warp ms avg/max: " << stage->warp_ms_avg << " / " << stage->warp_ms_max << "\n";
        out << "- Infer ms avg/max: " << stage->infer_ms_avg << " / " << stage->infer_ms_max << "\n";
        out << "- Total ms avg/max: " << stage->total_ms_avg << " / " << stage->total_ms_max << "\n";
        out << "- Trigger count: " << stage->trigger_count << "\n";
        out << "- Last red mean / red score: " << stage->last_red_mean << " / " << stage->last_red_score << "\n\n";
    }

    if (lock && lock->valid) {
        out << "## Calibration\n";
        out << "- Family: `" << lock->family << "`\n";
        out << "- Tag ID: `" << lock->id << "`\n";
        out << "- Source size: `" << lock->source_size.width << 'x' << lock->source_size.height << "`\n";
        out << "- Tag rect size: `" << lock->tag_rect_size.width << 'x' << lock->tag_rect_size.height << "`\n";
        out << "- Warp preview size: `" << lock->warp_size.width << 'x' << lock->warp_size.height << "`\n\n";
    }

    if (rois) {
        out << "## ROIs\n";
        out << "- red_roi: `" << rois->red_roi.x << ',' << rois->red_roi.y << ',' << rois->red_roi.w << ',' << rois->red_roi.h << "`\n";
        out << "- image_roi: `" << rois->image_roi.x << ',' << rois->image_roi.y << ',' << rois->image_roi.w << ',' << rois->image_roi.h << "`\n";
    }

    return true;
}

} // namespace vision_app
