#include "config.h"
#include "camera.h"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace app {

static void to_json(json& j, const RoiRatio& r) {
    j = json{{"x", r.x}, {"y", r.y}, {"w", r.w}, {"h", r.h}};
}

static void from_json(const json& j, RoiRatio& r) {
    j.at("x").get_to(r.x);
    j.at("y").get_to(r.y);
    j.at("w").get_to(r.w);
    j.at("h").get_to(r.h);
}

AppConfig ProfileStore::makeDefault() {
    AppConfig cfg;
    cfg.roi.red_roi = {0.10, 0.10, 0.20, 0.12};
    cfg.roi.target_roi = {0.30, 0.30, 0.35, 0.25};
    return cfg;
}

bool ProfileStore::save(const std::string& path, const AppConfig& cfg, std::string* err) {
    try {
        ensureParentDir(path);
        json j;
        j["profile_name"] = cfg.profile_name;
        j["camera"] = {
            {"probe_report_path", cfg.camera.probe_report_path},
            {"device_index", cfg.camera.device_index},
            {"width", cfg.camera.width},
            {"height", cfg.camera.height},
            {"fps", cfg.camera.fps},
            {"fourcc", cfg.camera.fourcc},
            {"backend", cfg.camera.backend},
            {"buffer_size", cfg.camera.buffer_size},
            {"flip_horizontal", cfg.camera.flip_horizontal},
            {"flip_vertical", cfg.camera.flip_vertical},
            {"actual_width", cfg.camera.actual_width},
            {"actual_height", cfg.camera.actual_height},
            {"actual_fps", cfg.camera.actual_fps},
            {"actual_fourcc", cfg.camera.actual_fourcc}
        };
        j["probe"] = {
            {"csv_path", cfg.probe.csv_path},
            {"json_path", cfg.probe.json_path}
        };
        j["remap"] = {
            {"tag_family", cfg.remap.tag_family},
            {"tag_orientation", cfg.remap.tag_orientation},
            {"homography_matrix", cfg.remap.homography_matrix},
            {"transformed_width", cfg.remap.transformed_width},
            {"transformed_height", cfg.remap.transformed_height},
            {"calibrated", cfg.remap.calibrated}
        };
        j["roi"] = {
            {"red_roi", cfg.roi.red_roi},
            {"red_threshold", cfg.roi.red_threshold},
            {"red_margin", cfg.roi.red_margin},
            {"cooldown_ms", cfg.roi.cooldown_ms},
            {"target_roi", cfg.roi.target_roi}
        };
        j["model"] = {
            {"backend", cfg.model.backend},
            {"model_path", cfg.model.model_path},
            {"confidence_threshold", cfg.model.confidence_threshold}
        };
        j["debug"] = {
            {"enable_ui", cfg.debug.enable_ui},
            {"manual_warp_preview", cfg.debug.manual_warp_preview},
            {"save_captures", cfg.debug.save_captures},
            {"verbose_log", cfg.debug.verbose_log}
        };

        std::ofstream ofs(path);
        ofs << j.dump(2);
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

bool ProfileStore::load(const std::string& path, AppConfig& cfg, std::string* err) {
    try {
        std::ifstream ifs(path);
        if (!ifs.is_open()) {
            if (err) *err = "cannot open config file";
            return false;
        }
        json j; ifs >> j;
        cfg = makeDefault();

        cfg.profile_name = j.value("profile_name", cfg.profile_name);
        auto jc = j.at("camera");
        cfg.camera.probe_report_path = jc.value("probe_report_path", cfg.camera.probe_report_path);
        cfg.camera.device_index = jc.value("device_index", cfg.camera.device_index);
        cfg.camera.width = jc.value("width", cfg.camera.width);
        cfg.camera.height = jc.value("height", cfg.camera.height);
        cfg.camera.fps = jc.value("fps", cfg.camera.fps);
        cfg.camera.fourcc = jc.value("fourcc", cfg.camera.fourcc);
        cfg.camera.backend = jc.value("backend", cfg.camera.backend);
        cfg.camera.buffer_size = jc.value("buffer_size", cfg.camera.buffer_size);
        cfg.camera.flip_horizontal = jc.value("flip_horizontal", cfg.camera.flip_horizontal);
        cfg.camera.flip_vertical = jc.value("flip_vertical", cfg.camera.flip_vertical);
        cfg.camera.actual_width = jc.value("actual_width", cfg.camera.actual_width);
        cfg.camera.actual_height = jc.value("actual_height", cfg.camera.actual_height);
        cfg.camera.actual_fps = jc.value("actual_fps", cfg.camera.actual_fps);
        cfg.camera.actual_fourcc = jc.value("actual_fourcc", cfg.camera.actual_fourcc);

        if (j.contains("probe")) {
            auto jp = j.at("probe");
            cfg.probe.csv_path = jp.value("csv_path", cfg.probe.csv_path);
            cfg.probe.json_path = jp.value("json_path", cfg.probe.json_path);
        }

        auto jr = j.at("remap");
        cfg.remap.tag_family = jr.value("tag_family", cfg.remap.tag_family);
        cfg.remap.tag_orientation = jr.value("tag_orientation", cfg.remap.tag_orientation);
        cfg.remap.homography_matrix = jr.value("homography_matrix", cfg.remap.homography_matrix);
        cfg.remap.transformed_width = jr.value("transformed_width", cfg.remap.transformed_width);
        cfg.remap.transformed_height = jr.value("transformed_height", cfg.remap.transformed_height);
        cfg.remap.calibrated = jr.value("calibrated", cfg.remap.calibrated);

        auto jroi = j.at("roi");
        cfg.roi.red_roi = jroi.at("red_roi").get<RoiRatio>();
        cfg.roi.red_threshold = jroi.value("red_threshold", cfg.roi.red_threshold);
        cfg.roi.red_margin = jroi.value("red_margin", cfg.roi.red_margin);
        cfg.roi.cooldown_ms = jroi.value("cooldown_ms", cfg.roi.cooldown_ms);
        cfg.roi.target_roi = jroi.at("target_roi").get<RoiRatio>();

        auto jm = j.at("model");
        cfg.model.backend = jm.value("backend", cfg.model.backend);
        cfg.model.model_path = jm.value("model_path", cfg.model.model_path);
        cfg.model.confidence_threshold = jm.value("confidence_threshold", cfg.model.confidence_threshold);

        if (j.contains("debug")) {
            auto jd = j.at("debug");
            cfg.debug.enable_ui = jd.value("enable_ui", cfg.debug.enable_ui);
            cfg.debug.manual_warp_preview = jd.value("manual_warp_preview", cfg.debug.manual_warp_preview);
            cfg.debug.save_captures = jd.value("save_captures", cfg.debug.save_captures);
            cfg.debug.verbose_log = jd.value("verbose_log", cfg.debug.verbose_log);
        }

        return validate(cfg, err);
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

bool ProfileStore::validate(const AppConfig& cfg, std::string* err) {
    if (cfg.camera.width <= 0 || cfg.camera.height <= 0 || cfg.camera.fps <= 0) {
        if (err) *err = "invalid requested camera mode";
        return false;
    }
    if (cfg.roi.red_threshold < 0 || cfg.roi.red_threshold > 255) {
        if (err) *err = "red threshold must be within [0,255]";
        return false;
    }
    auto check_ratio = [&](const RoiRatio& r, const char* name) {
        if (r.x < 0 || r.y < 0 || r.w <= 0 || r.h <= 0 || r.x + r.w > 1.0 || r.y + r.h > 1.0) {
            if (err) *err = std::string("invalid ratio roi: ") + name;
            return false;
        }
        return true;
    };
    return check_ratio(cfg.roi.red_roi, "red_roi") && check_ratio(cfg.roi.target_roi, "target_roi");
}

} // namespace app
