#include "camera.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace
{
std::string make_timestamp_string()
{
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
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

void apply_camera_settings(cv::VideoCapture& cap, const CameraConfig& cfg)
{
    if (cfg.prefer_mjpg)
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    if (cfg.buffer_size > 0)
        cap.set(cv::CAP_PROP_BUFFERSIZE, cfg.buffer_size);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);
    cap.set(cv::CAP_PROP_FPS, cfg.fps);
}

ProbeResult probe_one(int index, int backend, int width, int height, int fps, int warmup_frames, int measure_frames)
{
    ProbeResult result;
    result.camera_index = index;
    result.backend = backend;
    result.requested_width = width;
    result.requested_height = height;
    result.requested_fps = fps;

    cv::VideoCapture cap(index, backend);
    if (!cap.isOpened())
    {
        result.note = "open_failed";
        return result;
    }

    result.opened = true;
    CameraConfig cfg;
    cfg.index = index;
    cfg.width = width;
    cfg.height = height;
    cfg.fps = fps;
    cfg.backend = backend;
    cfg.prefer_mjpg = true;
    cfg.buffer_size = 1;
    apply_camera_settings(cap, cfg);

    result.actual_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    result.actual_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    result.actual_fps_property = cap.get(cv::CAP_PROP_FPS);

    cv::Mat frame;
    for (int i = 0; i < warmup_frames; ++i)
    {
        if (!cap.read(frame) || frame.empty())
        {
            result.note = "warmup_read_failed";
            return result;
        }
    }

    int good_frames = 0;
    double luma_sum = 0.0;
    double luma_sq_sum = 0.0;

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < measure_frames; ++i)
    {
        if (!cap.read(frame) || frame.empty())
        {
            result.note = "measure_read_failed";
            break;
        }

        cv::Mat gray;
        if (frame.channels() == 3)
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        else
            gray = frame;

        cv::Scalar mean, stddev;
        cv::meanStdDev(gray, mean, stddev);
        luma_sum += mean[0];
        luma_sq_sum += stddev[0];
        ++good_frames;
    }
    const auto t1 = std::chrono::steady_clock::now();

    if (good_frames > 0)
    {
        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        if (seconds > 0.0)
            result.measured_fps = static_cast<double>(good_frames) / seconds;
        result.mean_luma = luma_sum / static_cast<double>(good_frames);
        result.stddev_luma = luma_sq_sum / static_cast<double>(good_frames);
        result.stable = (good_frames == measure_frames);
        if (result.note.empty())
            result.note = result.stable ? "ok" : "partial";
    }
    else if (result.note.empty())
    {
        result.note = "no_frames";
    }

    return result;
}
}

bool open_camera(cv::VideoCapture& cap, const CameraConfig& cfg, std::string* error)
{
    cap.open(cfg.index, cfg.backend);
    if (!cap.isOpened())
    {
        if (error) *error = "Failed to open camera index " + std::to_string(cfg.index) + " with " + backend_to_string(cfg.backend);
        return false;
    }

    apply_camera_settings(cap, cfg);

    cv::Mat frame;
    for (int i = 0; i < cfg.warmup_frames; ++i)
    {
        if (!cap.read(frame) || frame.empty())
        {
            if (error) *error = "Camera opened but warmup read failed";
            return false;
        }
    }
    return true;
}

bool read_frame(cv::VideoCapture& cap, cv::Mat& frame, int drop_frames_per_read)
{
    for (int i = 0; i < drop_frames_per_read; ++i)
    {
        if (!cap.grab())
            return false;
    }
    return cap.read(frame) && !frame.empty();
}

std::vector<ProbeResult> run_camera_probe(const AppConfig& cfg)
{
    std::vector<ProbeResult> results;
    for (int index : cfg.probe.camera_indices)
    {
        for (int backend : cfg.probe.backends)
        {
            const size_t n = std::min(cfg.probe.widths.size(), cfg.probe.heights.size());
            for (size_t i = 0; i < n; ++i)
            {
                for (int fps : cfg.probe.fps_values)
                {
                    results.push_back(
                        probe_one(index,
                                  backend,
                                  cfg.probe.widths[i],
                                  cfg.probe.heights[i],
                                  fps,
                                  cfg.probe.warmup_frames,
                                  cfg.probe.measure_frames));
                }
            }
        }
    }
    return results;
}

bool write_probe_report(const std::string& report_dir, const std::vector<ProbeResult>& results, std::string* json_path, std::string* csv_path)
{
    try
    {
        fs::create_directories(report_dir);
        const std::string stamp = make_timestamp_string();
        const fs::path json_file = fs::path(report_dir) / ("camera_probe_" + stamp + ".json");
        const fs::path csv_file = fs::path(report_dir) / ("camera_probe_" + stamp + ".csv");

        json j = json::array();
        for (const auto& r : results)
        {
            j.push_back({
                {"camera_index", r.camera_index},
                {"backend", r.backend},
                {"backend_name", backend_to_string(r.backend)},
                {"requested_width", r.requested_width},
                {"requested_height", r.requested_height},
                {"requested_fps", r.requested_fps},
                {"actual_width", r.actual_width},
                {"actual_height", r.actual_height},
                {"actual_fps_property", r.actual_fps_property},
                {"measured_fps", r.measured_fps},
                {"opened", r.opened},
                {"stable", r.stable},
                {"mean_luma", r.mean_luma},
                {"stddev_luma", r.stddev_luma},
                {"note", r.note}
            });
        }

        std::ofstream jout(json_file);
        jout << j.dump(2) << '\n';

        std::ofstream coutf(csv_file);
        coutf << "camera_index,backend,requested_width,requested_height,requested_fps,actual_width,actual_height,actual_fps_property,measured_fps,opened,stable,mean_luma,stddev_luma,note\n";
        for (const auto& r : results)
        {
            coutf << r.camera_index << ','
                  << backend_to_string(r.backend) << ','
                  << r.requested_width << ','
                  << r.requested_height << ','
                  << r.requested_fps << ','
                  << r.actual_width << ','
                  << r.actual_height << ','
                  << r.actual_fps_property << ','
                  << r.measured_fps << ','
                  << (r.opened ? 1 : 0) << ','
                  << (r.stable ? 1 : 0) << ','
                  << r.mean_luma << ','
                  << r.stddev_luma << ','
                  << r.note << '\n';
        }

        if (json_path) *json_path = json_file.string();
        if (csv_path) *csv_path = csv_file.string();
        return true;
    }
    catch (...)
    {
        return false;
    }
}
