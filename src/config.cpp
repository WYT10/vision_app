#include "config.h"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <unordered_map>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace
{
json roi_to_json(const RoiRatio& roi)
{
    return json{{"x", roi.x}, {"y", roi.y}, {"w", roi.w}, {"h", roi.h}};
}

RoiRatio roi_from_json(const json& j)
{
    RoiRatio roi;
    roi.x = j.value("x", 0.0);
    roi.y = j.value("y", 0.0);
    roi.w = j.value("w", 0.0);
    roi.h = j.value("h", 0.0);
    return roi;
}

json mode_to_json(const CameraMode& mode)
{
    return json{{"width", mode.width}, {"height", mode.height}, {"fps", mode.fps}, {"fourcc", mode.fourcc}};
}

CameraMode mode_from_json(const json& j, const CameraMode& defaults)
{
    CameraMode mode = defaults;
    mode.width = j.value("width", mode.width);
    mode.height = j.value("height", mode.height);
    mode.fps = j.value("fps", mode.fps);
    mode.fourcc = normalize_fourcc_string(j.value("fourcc", mode.fourcc));
    return mode;
}

json homography_to_json(const cv::Mat& H)
{
    json arr = json::array();
    if (H.empty())
        return arr;

    cv::Mat H64;
    H.convertTo(H64, CV_64F);
    for (int r = 0; r < H64.rows; ++r)
        for (int c = 0; c < H64.cols; ++c)
            arr.push_back(H64.at<double>(r, c));
    return arr;
}

cv::Mat homography_from_json(const json& j)
{
    if (!j.is_array() || j.size() != 9)
        return {};

    cv::Mat H(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i)
        H.at<double>(i / 3, i % 3) = j.at(i).get<double>();
    return H;
}
}

std::string normalize_fourcc_string(const std::string& fourcc)
{
    if (fourcc.empty())
        return "MJPG";

    std::string out = fourcc;
    for (char& ch : out)
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    if (out.size() > 4)
        out.resize(4);
    while (out.size() < 4)
        out.push_back(' ');
    return out;
}

std::string backend_to_string(int backend)
{
    switch (backend)
    {
        case cv::CAP_ANY: return "ANY";
        case cv::CAP_V4L2: return "V4L2";
        case cv::CAP_GSTREAMER: return "GSTREAMER";
        case cv::CAP_FFMPEG: return "FFMPEG";
        case cv::CAP_DSHOW: return "DSHOW";
        case cv::CAP_MSMF: return "MSMF";
        default: return std::to_string(backend);
    }
}

int backend_from_string(const std::string& backend_name)
{
    static const std::unordered_map<std::string, int> table = {
        {"ANY", cv::CAP_ANY},
        {"V4L2", cv::CAP_V4L2},
        {"GSTREAMER", cv::CAP_GSTREAMER},
        {"FFMPEG", cv::CAP_FFMPEG},
        {"DSHOW", cv::CAP_DSHOW},
        {"MSMF", cv::CAP_MSMF}
    };

    auto it = table.find(backend_name);
    if (it != table.end())
        return it->second;

    try
    {
        return std::stoi(backend_name);
    }
    catch (...)
    {
        return cv::CAP_V4L2;
    }
}

bool camera_modes_match(const CameraMode& a, const CameraMode& b)
{
    if (a.width != b.width || a.height != b.height)
        return false;

    const bool fps_known = a.fps > 0 && b.fps > 0;
    if (fps_known && a.fps != b.fps)
        return false;

    return normalize_fourcc_string(a.fourcc) == normalize_fourcc_string(b.fourcc);
}

bool load_config(const std::string& path, AppConfig& config, std::string* error)
{
    try
    {
        std::ifstream in(path);
        if (!in.is_open())
        {
            if (error) *error = "Failed to open config: " + path;
            return false;
        }

        json j;
        in >> j;

        const auto& jc = j.value("camera", json::object());
        config.camera.device_index = jc.value("device_index", config.camera.device_index);
        config.camera.device_path = jc.value("device_path", config.camera.device_path);
        config.camera.backend = backend_from_string(jc.value("backend", backend_to_string(config.camera.backend)));
        config.camera.requested_mode = mode_from_json(jc.value("requested_mode", json::object()), config.camera.requested_mode);
        config.camera.buffer_size = jc.value("buffer_size", config.camera.buffer_size);
        config.camera.warmup_frames = jc.value("warmup_frames", config.camera.warmup_frames);
        config.camera.drop_frames_per_read = jc.value("drop_frames_per_read", config.camera.drop_frames_per_read);
        config.camera.flip_horizontal = jc.value("flip_horizontal", config.camera.flip_horizontal);

        config.camera.probe_candidates.clear();
        if (jc.contains("probe_candidates") && jc["probe_candidates"].is_array())
        {
            for (const auto& item : jc["probe_candidates"])
                config.camera.probe_candidates.push_back(mode_from_json(item, config.camera.requested_mode));
        }
        if (config.camera.probe_candidates.empty())
            config.camera.probe_candidates.push_back(config.camera.requested_mode);

        const auto& jt = j.value("tag", json::object());
        config.tag.family_mode = jt.value("family_mode", config.tag.family_mode);
        config.tag.allowed_family = jt.value("allowed_family", config.tag.allowed_family);
        config.tag.allowed_id = jt.value("allowed_id", config.tag.allowed_id);
        config.tag.tag_size_units = jt.value("tag_size_units", config.tag.tag_size_units);
        config.tag.output_padding_units = jt.value("output_padding_units", config.tag.output_padding_units);
        config.tag.lock_on_first_detection = jt.value("lock_on_first_detection", config.tag.lock_on_first_detection);

        const auto& jtr = j.value("trigger", json::object());
        config.trigger.red_threshold = jtr.value("red_threshold", config.trigger.red_threshold);
        config.trigger.red_margin = jtr.value("red_margin", config.trigger.red_margin);
        config.trigger.cooldown_ms = jtr.value("cooldown_ms", config.trigger.cooldown_ms);
        config.trigger.save_raw = jtr.value("save_raw", config.trigger.save_raw);
        config.trigger.save_warped = jtr.value("save_warped", config.trigger.save_warped);
        config.trigger.save_roi = jtr.value("save_roi", config.trigger.save_roi);
        config.trigger.capture_dir = jtr.value("capture_dir", config.trigger.capture_dir);

        const auto& jr = j.value("runtime", json::object());
        config.runtime.show_ui = jr.value("show_ui", config.runtime.show_ui);
        config.runtime.headless_deploy = jr.value("headless_deploy", config.runtime.headless_deploy);
        config.runtime.probe_measure_frames = jr.value("probe_measure_frames", config.runtime.probe_measure_frames);
        config.runtime.report_dir = jr.value("report_dir", config.runtime.report_dir);

        const auto& jcal = j.value("calibration", json::object());
        config.calibration.valid = jcal.value("valid", config.calibration.valid);
        config.calibration.camera_mode_used = mode_from_json(jcal.value("camera_mode_used", json::object()), config.calibration.camera_mode_used);
        config.calibration.warped_width = jcal.value("warped_width", config.calibration.warped_width);
        config.calibration.warped_height = jcal.value("warped_height", config.calibration.warped_height);
        config.calibration.red_roi_ratio = roi_from_json(jcal.value("red_roi_ratio", json::object()));
        config.calibration.image_roi_ratio = roi_from_json(jcal.value("image_roi_ratio", json::object()));
        config.calibration.homography = homography_from_json(jcal.value("homography", json::array()));

        if (config.calibration.valid &&
            (config.calibration.homography.empty() ||
             config.calibration.warped_width <= 0 ||
             config.calibration.warped_height <= 0))
        {
            config.calibration.valid = false;
        }

        return true;
    }
    catch (const std::exception& e)
    {
        if (error) *error = e.what();
        return false;
    }
}

bool save_config(const std::string& path, const AppConfig& config, std::string* error)
{
    try
    {
        json j;
        j["camera"] = {
            {"device_index", config.camera.device_index},
            {"device_path", config.camera.device_path},
            {"backend", backend_to_string(config.camera.backend)},
            {"requested_mode", mode_to_json(config.camera.requested_mode)},
            {"buffer_size", config.camera.buffer_size},
            {"warmup_frames", config.camera.warmup_frames},
            {"drop_frames_per_read", config.camera.drop_frames_per_read},
            {"flip_horizontal", config.camera.flip_horizontal},
            {"probe_candidates", json::array()}
        };
        for (const auto& mode : config.camera.probe_candidates)
            j["camera"]["probe_candidates"].push_back(mode_to_json(mode));

        j["tag"] = {
            {"family_mode", config.tag.family_mode},
            {"allowed_family", config.tag.allowed_family},
            {"allowed_id", config.tag.allowed_id},
            {"tag_size_units", config.tag.tag_size_units},
            {"output_padding_units", config.tag.output_padding_units},
            {"lock_on_first_detection", config.tag.lock_on_first_detection}
        };

        j["trigger"] = {
            {"red_threshold", config.trigger.red_threshold},
            {"red_margin", config.trigger.red_margin},
            {"cooldown_ms", config.trigger.cooldown_ms},
            {"save_raw", config.trigger.save_raw},
            {"save_warped", config.trigger.save_warped},
            {"save_roi", config.trigger.save_roi},
            {"capture_dir", config.trigger.capture_dir}
        };

        j["runtime"] = {
            {"show_ui", config.runtime.show_ui},
            {"headless_deploy", config.runtime.headless_deploy},
            {"probe_measure_frames", config.runtime.probe_measure_frames},
            {"report_dir", config.runtime.report_dir}
        };

        j["calibration"] = {
            {"valid", config.calibration.valid},
            {"camera_mode_used", mode_to_json(config.calibration.camera_mode_used)},
            {"warped_width", config.calibration.warped_width},
            {"warped_height", config.calibration.warped_height},
            {"homography", homography_to_json(config.calibration.homography)},
            {"red_roi_ratio", roi_to_json(config.calibration.red_roi_ratio)},
            {"image_roi_ratio", roi_to_json(config.calibration.image_roi_ratio)}
        };

        fs::path out_path(path);
        if (!out_path.parent_path().empty())
            fs::create_directories(out_path.parent_path());

        std::ofstream out(path);
        if (!out.is_open())
        {
            if (error) *error = "Failed to write config: " + path;
            return false;
        }
        out << j.dump(2) << '\n';
        return true;
    }
    catch (const std::exception& e)
    {
        if (error) *error = e.what();
        return false;
    }
}
