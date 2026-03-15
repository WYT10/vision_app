#include "camera.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace
{
/*
------------------------------------------------------------------------------
Timestamp helper for probe reports and saved captures.
------------------------------------------------------------------------------
*/
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

/* Convert FOURCC string <-> backend integer code. */
int fourcc_to_int(const std::string& fourcc)
{
    const std::string f = normalize_fourcc_string(fourcc);
    return cv::VideoWriter::fourcc(f[0], f[1], f[2], f[3]);
}

std::string int_to_fourcc(double value)
{
    const int code = static_cast<int>(value);
    std::string out(4, ' ');
    out[0] = static_cast<char>(code & 0xFF);
    out[1] = static_cast<char>((code >> 8) & 0xFF);
    out[2] = static_cast<char>((code >> 16) & 0xFF);
    out[3] = static_cast<char>((code >> 24) & 0xFF);
    return normalize_fourcc_string(out);
}

/*
------------------------------------------------------------------------------
V4L2 helper selection
------------------------------------------------------------------------------
For USB cameras on Linux, the most stable node for the `v4l2-ctl` path is often
/dev/videoN. If the config already provides a node, use it. Otherwise derive it
from the camera index.
------------------------------------------------------------------------------
*/
std::string choose_v4l2_device(const CameraConfig& cfg)
{
    if (!cfg.v4l2_device.empty())
        return cfg.v4l2_device;

    if (!cfg.device_path.empty() && cfg.device_path.rfind("/dev/video", 0) == 0)
        return cfg.device_path;

    if (cfg.device_index >= 0)
        return "/dev/video" + std::to_string(cfg.device_index);

    return {};
}

/* Run a shell command and capture stdout+stderr. */
std::string run_command_capture(const std::string& command)
{
    std::array<char, 512> buffer{};
    std::string output;

    std::string command_with_stderr = command + " 2>&1";
    FILE* pipe = popen(command_with_stderr.c_str(), "r");
    if (!pipe)
        return "[probe] failed to launch command: " + command + "\n";

    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr)
        output += buffer.data();

    const int rc = pclose(pipe);
    if (rc != 0 && output.empty())
        output = "[probe] command returned non-zero exit code\n";
    return output;
}

/*
------------------------------------------------------------------------------
Camera setting application
------------------------------------------------------------------------------
Order matters slightly on real hardware. FOURCC first often gives more stable
results with USB cameras because the driver chooses a transport mode earlier.
------------------------------------------------------------------------------
*/
void apply_camera_settings(cv::VideoCapture& cap, const CameraConfig& cfg, const CameraMode& mode)
{
    cap.set(cv::CAP_PROP_FOURCC, fourcc_to_int(mode.fourcc));
    if (cfg.buffer_size > 0)
        cap.set(cv::CAP_PROP_BUFFERSIZE, cfg.buffer_size);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, mode.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, mode.height);
    cap.set(cv::CAP_PROP_FPS, mode.fps);
}

CameraMode query_actual_mode(cv::VideoCapture& cap)
{
    CameraMode out;
    out.width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    out.height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    out.fps = static_cast<int>(std::round(cap.get(cv::CAP_PROP_FPS)));
    out.fourcc = int_to_fourcc(cap.get(cv::CAP_PROP_FOURCC));
    return out;
}

/*
------------------------------------------------------------------------------
Single candidate probe
------------------------------------------------------------------------------
This is the real-time test layer.

Input
    camera_cfg     : camera open parameters
    mode           : requested candidate mode
    measure_frames : number of frames used for FPS measurement

Output
    ProbeResult containing requested mode, actual mode, stability, measured FPS,
    and simple image statistics.
------------------------------------------------------------------------------
*/
ProbeResult probe_one(const CameraConfig& camera_cfg, const CameraMode& mode, int measure_frames)
{
    ProbeResult result;
    result.camera_index = camera_cfg.device_index;
    result.device_path = camera_cfg.device_path;
    result.backend = camera_cfg.backend;
    result.requested_mode = mode;

    cv::VideoCapture cap;
    const bool use_path = !camera_cfg.device_path.empty();
    if (use_path)
        cap.open(camera_cfg.device_path, camera_cfg.backend);
    else
        cap.open(camera_cfg.device_index, camera_cfg.backend);

    if (!cap.isOpened())
    {
        result.note = "open_failed";
        return result;
    }

    result.opened = true;
    apply_camera_settings(cap, camera_cfg, mode);
    result.actual_mode = query_actual_mode(cap);

    cv::Mat frame;
    for (int i = 0; i < camera_cfg.warmup_frames; ++i)
    {
        if (!cap.read(frame) || frame.empty())
        {
            result.note = "warmup_read_failed";
            return result;
        }
    }

    int good_frames = 0;
    double mean_sum = 0.0;
    double std_sum = 0.0;

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
        mean_sum += mean[0];
        std_sum += stddev[0];
        ++good_frames;
    }
    const auto t1 = std::chrono::steady_clock::now();

    if (good_frames > 0)
    {
        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        if (seconds > 0.0)
            result.measured_fps = static_cast<double>(good_frames) / seconds;
        result.mean_luma = mean_sum / static_cast<double>(good_frames);
        result.stddev_luma = std_sum / static_cast<double>(good_frames);
        result.stable = (good_frames == measure_frames);
    }

    if (result.note.empty())
    {
        if (!camera_modes_match(result.actual_mode, mode))
            result.note = "mode_mismatch";
        else if (result.stable)
            result.note = "ok";
        else
            result.note = "partial";
    }

    return result;
}

json probe_result_to_json(const ProbeResult& r)
{
    return {
        {"camera_index", r.camera_index},
        {"device_path", r.device_path},
        {"backend", backend_to_string(r.backend)},
        {"requested_mode", {
            {"width", r.requested_mode.width},
            {"height", r.requested_mode.height},
            {"fps", r.requested_mode.fps},
            {"fourcc", r.requested_mode.fourcc}
        }},
        {"actual_mode", {
            {"width", r.actual_mode.width},
            {"height", r.actual_mode.height},
            {"fps", r.actual_mode.fps},
            {"fourcc", r.actual_mode.fourcc}
        }},
        {"opened", r.opened},
        {"stable", r.stable},
        {"measured_fps", r.measured_fps},
        {"mean_luma", r.mean_luma},
        {"stddev_luma", r.stddev_luma},
        {"note", r.note}
    };
}
}

bool open_camera(cv::VideoCapture& cap, const CameraConfig& cfg, CameraMode* actual_mode, std::string* error)
{
    const bool use_path = !cfg.device_path.empty();
    if (use_path)
        cap.open(cfg.device_path, cfg.backend);
    else
        cap.open(cfg.device_index, cfg.backend);

    if (!cap.isOpened())
    {
        if (error)
        {
            std::ostringstream oss;
            oss << "Failed to open camera "
                << (use_path ? cfg.device_path : std::to_string(cfg.device_index))
                << " with backend " << backend_to_string(cfg.backend);
            *error = oss.str();
        }
        return false;
    }

    apply_camera_settings(cap, cfg, cfg.requested_mode);

    cv::Mat frame;
    for (int i = 0; i < cfg.warmup_frames; ++i)
    {
        if (!cap.read(frame) || frame.empty())
        {
            if (error) *error = "Camera opened but warmup read failed";
            return false;
        }
    }

    if (actual_mode)
        *actual_mode = query_actual_mode(cap);
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

ProbeReport run_camera_probe(const AppConfig& cfg)
{
    ProbeReport report;
    report.v4l2_device = choose_v4l2_device(cfg.camera);

    std::cout << "[probe] backend=" << backend_to_string(cfg.camera.backend)
              << " camera=" << (cfg.camera.device_path.empty() ? std::to_string(cfg.camera.device_index) : cfg.camera.device_path)
              << " requested_active_mode=" << cfg.camera.requested_mode.width << 'x' << cfg.camera.requested_mode.height
              << '@' << cfg.camera.requested_mode.fps << ' ' << normalize_fourcc_string(cfg.camera.requested_mode.fourcc) << '\n';

    if (!report.v4l2_device.empty())
    {
        std::cout << "[probe] collecting V4L2 driver enumeration from " << report.v4l2_device << " ...\n";
        report.list_devices_output = run_command_capture("v4l2-ctl --list-devices");
        report.list_formats_output = run_command_capture("v4l2-ctl -d " + report.v4l2_device + " --list-formats-ext");
    }
    else
    {
        report.list_devices_output = "[probe] no V4L2 device node available\n";
        report.list_formats_output = "[probe] no V4L2 device node available\n";
    }

    report.results.reserve(cfg.camera.probe_candidates.size());

    for (size_t i = 0; i < cfg.camera.probe_candidates.size(); ++i)
    {
        const auto& mode = cfg.camera.probe_candidates[i];
        std::cout << "[probe] [" << (i + 1) << '/' << cfg.camera.probe_candidates.size() << "] testing "
                  << mode.width << 'x' << mode.height << '@' << mode.fps << ' ' << normalize_fourcc_string(mode.fourcc) << " ... " << std::flush;

        ProbeResult result = probe_one(cfg.camera, mode, cfg.runtime.probe_measure_frames);
        report.results.push_back(result);

        std::cout << result.note
                  << " | actual=" << result.actual_mode.width << 'x' << result.actual_mode.height
                  << '@' << result.actual_mode.fps << ' ' << normalize_fourcc_string(result.actual_mode.fourcc)
                  << " | measured_fps=" << std::fixed << std::setprecision(2) << result.measured_fps
                  << '\n';
    }

    std::cout << "[probe] done\n";
    return report;
}

bool write_probe_report(const std::string& report_dir, const ProbeReport& report, std::string* json_path, std::string* csv_path, std::string* txt_path)
{
    try
    {
        fs::create_directories(report_dir);
        const std::string stamp = make_timestamp_string();
        const fs::path json_file = fs::path(report_dir) / ("camera_probe_" + stamp + ".json");
        const fs::path csv_file = fs::path(report_dir) / ("camera_probe_" + stamp + ".csv");
        const fs::path txt_file = fs::path(report_dir) / ("camera_probe_" + stamp + "_v4l2.txt");

        json j;
        j["v4l2_device"] = report.v4l2_device;
        j["v4l2_list_devices_raw"] = report.list_devices_output;
        j["v4l2_list_formats_ext_raw"] = report.list_formats_output;
        j["probe_results"] = json::array();
        for (const auto& r : report.results)
            j["probe_results"].push_back(probe_result_to_json(r));

        std::ofstream jout(json_file);
        if (!jout.is_open())
            return false;
        jout << j.dump(2) << '\n';

        std::ofstream coutf(csv_file);
        if (!coutf.is_open())
            return false;
        coutf << "camera_index,backend,req_width,req_height,req_fps,req_fourcc,act_width,act_height,act_fps,act_fourcc,opened,stable,measured_fps,mean_luma,stddev_luma,note\n";
        for (const auto& r : report.results)
        {
            coutf << r.camera_index << ','
                  << backend_to_string(r.backend) << ','
                  << r.requested_mode.width << ','
                  << r.requested_mode.height << ','
                  << r.requested_mode.fps << ','
                  << r.requested_mode.fourcc << ','
                  << r.actual_mode.width << ','
                  << r.actual_mode.height << ','
                  << r.actual_mode.fps << ','
                  << r.actual_mode.fourcc << ','
                  << (r.opened ? 1 : 0) << ','
                  << (r.stable ? 1 : 0) << ','
                  << r.measured_fps << ','
                  << r.mean_luma << ','
                  << r.stddev_luma << ','
                  << r.note << '\n';
        }

        std::ofstream tout(txt_file);
        if (!tout.is_open())
            return false;
        tout << "==== v4l2-ctl --list-devices ====\n";
        tout << report.list_devices_output << '\n';
        tout << "==== v4l2-ctl -d " << report.v4l2_device << " --list-formats-ext ====\n";
        tout << report.list_formats_output << '\n';

        if (json_path) *json_path = json_file.string();
        if (csv_path) *csv_path = csv_file.string();
        if (txt_path) *txt_path = txt_file.string();
        return true;
    }
    catch (...)
    {
        return false;
    }
}
