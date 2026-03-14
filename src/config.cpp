#include "config.h"

#include <fstream>
#include <sstream>
#include <filesystem>
#include <nlohmann/json.hpp>

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
        config.camera.index = jc.value("index", config.camera.index);
        config.camera.width = jc.value("width", config.camera.width);
        config.camera.height = jc.value("height", config.camera.height);
        config.camera.fps = jc.value("fps", config.camera.fps);
        config.camera.backend = jc.value("backend", config.camera.backend);
        config.camera.prefer_mjpg = jc.value("prefer_mjpg", config.camera.prefer_mjpg);
        config.camera.buffer_size = jc.value("buffer_size", config.camera.buffer_size);
        config.camera.warmup_frames = jc.value("warmup_frames", config.camera.warmup_frames);
        config.camera.drop_frames_per_read = jc.value("drop_frames_per_read", config.camera.drop_frames_per_read);
        config.camera.flip_horizontal = jc.value("flip_horizontal", config.camera.flip_horizontal);

        const auto& jp = j.value("probe", json::object());
        config.probe.camera_indices = jp.value("camera_indices", config.probe.camera_indices);
        config.probe.widths = jp.value("widths", config.probe.widths);
        config.probe.heights = jp.value("heights", config.probe.heights);
        config.probe.fps_values = jp.value("fps_values", config.probe.fps_values);
        config.probe.backends = jp.value("backends", config.probe.backends);
        config.probe.warmup_frames = jp.value("warmup_frames", config.probe.warmup_frames);
        config.probe.measure_frames = jp.value("measure_frames", config.probe.measure_frames);
        config.probe.report_dir = jp.value("report_dir", config.probe.report_dir);

        const auto& jt = j.value("tag", json::object());
        config.tag.family = jt.value("family", config.tag.family);
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

        const auto& jcal = j.value("calibration", json::object());
        config.calibration.valid = jcal.value("valid", config.calibration.valid);
        config.calibration.warped_width = jcal.value("warped_width", config.calibration.warped_width);
        config.calibration.warped_height = jcal.value("warped_height", config.calibration.warped_height);
        config.calibration.red_roi = roi_from_json(jcal.value("red_roi", json::object()));
        config.calibration.image_roi = roi_from_json(jcal.value("image_roi", json::object()));
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
            {"index", config.camera.index},
            {"width", config.camera.width},
            {"height", config.camera.height},
            {"fps", config.camera.fps},
            {"backend", config.camera.backend},
            {"prefer_mjpg", config.camera.prefer_mjpg},
            {"buffer_size", config.camera.buffer_size},
            {"warmup_frames", config.camera.warmup_frames},
            {"drop_frames_per_read", config.camera.drop_frames_per_read},
            {"flip_horizontal", config.camera.flip_horizontal}
        };

        j["probe"] = {
            {"camera_indices", config.probe.camera_indices},
            {"widths", config.probe.widths},
            {"heights", config.probe.heights},
            {"fps_values", config.probe.fps_values},
            {"backends", config.probe.backends},
            {"warmup_frames", config.probe.warmup_frames},
            {"measure_frames", config.probe.measure_frames},
            {"report_dir", config.probe.report_dir}
        };

        j["tag"] = {
            {"family", config.tag.family},
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
            {"headless_deploy", config.runtime.headless_deploy}
        };

        j["calibration"] = {
            {"valid", config.calibration.valid},
            {"warped_width", config.calibration.warped_width},
            {"warped_height", config.calibration.warped_height},
            {"homography", homography_to_json(config.calibration.homography)},
            {"red_roi", roi_to_json(config.calibration.red_roi)},
            {"image_roi", roi_to_json(config.calibration.image_roi)}
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

std::string backend_to_string(int backend)
{
    switch (backend)
    {
        case cv::CAP_ANY: return "CAP_ANY";
        case cv::CAP_V4L2: return "CAP_V4L2";
        case cv::CAP_GSTREAMER: return "CAP_GSTREAMER";
        case cv::CAP_FFMPEG: return "CAP_FFMPEG";
        case cv::CAP_DSHOW: return "CAP_DSHOW";
        case cv::CAP_MSMF: return "CAP_MSMF";
        default: return "BACKEND_" + std::to_string(backend);
    }
}
