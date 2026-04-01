#pragma once

#include <algorithm>
#include <chrono>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "camera.hpp"
#include "calibrate.hpp"
#include "model.hpp"
#include "stats.hpp"

namespace vision_app {

struct RedThresholdConfig {
    int h1_low = 0;
    int h1_high = 10;
    int h2_low = 170;
    int h2_high = 180;
    int s_min = 80;
    int v_min = 60;
};

struct DynamicRedRoiState {
    bool has_last_center = false;
    double filtered_center_x = 0.0;
    int miss_count = 0;
    void reset() {
        has_last_center = false;
        filtered_center_x = 0.0;
        miss_count = 0;
    }
};

bool extract_runtime_rois(const cv::Mat& warped,
                          const cv::Mat& valid_mask,
                          const std::string& roi_mode,
                          const RoiConfig& rois,
                          const RedThresholdConfig& red_cfg,
                          const DynamicRedRoiConfig& dynamic_cfg,
                          DynamicRedRoiState& dynamic_state,
                          RoiRuntimeData& out,
                          std::string& err);

struct AppOptions {
    std::string mode = "probe";
    std::string probe_task = "list"; // list | live | snap | bench
    std::string roi_mode = "fixed";  // fixed | dynamic-red-x
    std::string device = "/dev/video0";
    int width = 640;
    int height = 480;
    int fps = 120;
    std::string fourcc = "MJPG";
    int buffer_size = 1;
    bool latest_only = true;
    int drain_grabs = 1;
    bool ui = true;
    int duration = 10;
    bool draw_overlay = true;
    bool text_console = true;

    int camera_soft_max = 1000;
    int camera_preview_max = 640;
    int warp_preview_max = 640;

    int warp_width = 384;
    int warp_height = 384;
    int target_tag_px = 128;
    double warp_center_x_ratio = 0.5;
    double warp_center_y_ratio = 0.5;

    std::string tag_family = "auto";
    int target_id = 0;
    bool require_target_id = true;
    bool manual_lock_only = true;
    int lock_frames = 4;

    std::string save_warp = "../report/warp_package.yml.gz";
    std::string load_warp = "../report/warp_package.yml.gz";
    std::string save_rois = "../report/rois.yml";
    std::string load_rois = "../report/rois.yml";
    std::string save_report = "../report/latest_report.md";
    std::string snap_path;

    RoiConfig default_rois;
    DynamicRedRoiConfig dynamic_red;
    RedThresholdConfig red_cfg;
    ModelConfig model_cfg;
    double model_max_hz = 5.0;

    bool run_red = true;
    bool run_image_roi = true;
    bool run_model = true;

    std::string save_image_roi_dir;
    std::string save_red_roi_dir;
    int save_every_n = 0;
};

inline void print_calibrate_controls() {
    std::cout << R"TXT(
=== vision_app calibrate controls ===
Keys
  SPACE / ENTER : lock current tag
  u             : unlock / reacquire
  m             : toggle roi_mode fixed <-> dynamic-red-x
  1 / 2         : select red_roi / image_roi (fixed mode only)

Fixed mode tuning
  w a s d       : move ROI
  i / k         : height - / +
  j / l         : width  - / +
  [ / ]         : move step down / up
  , / .         : size step down / up

Dynamic-red-x tuning
  w / s         : move red band up / down
  i / k         : band height - / +
  a / d         : image ROI width - / +
  z / x         : image ROI height - / +
  j / l         : gap above band - / +
  n / b         : dual-zone mid gap - / +
  c / v         : zone min red pixels - / +
  f / g         : zone min red ratio - / +
  [ / ]         : dynamic pixel step down / up

General
  p             : save all
  y             : save warp only
  o             : save rois only
  r             : reset roi settings to defaults
  h             : show this help again
  q / ESC       : quit

Modes
  fixed         : classic saved red_roi + image_roi rectangles
  dynamic-red-x : detect red in a fixed band, estimate x-center, auto-place image roi above the band

Windows
  vision_app_camera   : raw feed + tag overlay
  vision_app_warp     : warp preview + active roi mode overlay
  vision_app_red_mask : optional dynamic red mask preview
)TXT" << std::flush;
}

inline cv::Mat make_blank_preview(int width, int height, const std::string& text) {
    cv::Mat img(std::max(64, height), std::max(64, width), CV_8UC3, cv::Scalar(235,235,235));
    cv::putText(img, text, {16, img.rows / 2}, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(60,60,60), 2);
    return img;
}

inline std::string build_deploy_config_warning(const AppOptions& opt,
                                               const WarpPackage& pack,
                                               int cam_w,
                                               int cam_h) {
    std::ostringstream oss;
    bool any = false;
    if (pack.src_size.width > 0 && pack.src_size.height > 0 &&
        (cam_w != pack.src_size.width || cam_h != pack.src_size.height)) {
        oss << "[warn] live camera size " << cam_w << 'x' << cam_h
            << " does not match saved warp source size "
            << pack.src_size.width << 'x' << pack.src_size.height
            << ". Deploy warp may be invalid until you use the same camera mode or recalibrate.\n";
        any = true;
    }
    if ((opt.warp_width > 0 && opt.warp_width != pack.warp_size.width) ||
        (opt.warp_height > 0 && opt.warp_height != pack.warp_size.height) ||
        (opt.target_tag_px > 0 && pack.target_tag_px > 0 && opt.target_tag_px != pack.target_tag_px)) {
        oss << "[warn] saved warp size/tag_px differ from current CLI hints:"
            << " saved=" << pack.warp_size.width << 'x' << pack.warp_size.height
            << " tag_px=" << pack.target_tag_px
            << " cli=" << opt.warp_width << 'x' << opt.warp_height
            << " tag_px=" << opt.target_tag_px << "\n";
        any = true;
    }
    return any ? oss.str() : std::string();
}

inline bool save_crop_if_needed(const std::string& dir,
                                const std::string& prefix,
                                int frame_idx,
                                const cv::Mat& img) {
    if (dir.empty() || img.empty()) return false;
    std::filesystem::create_directories(dir);
    std::ostringstream name;
    name << prefix << '_' << std::setw(6) << std::setfill('0') << frame_idx << ".jpg";
    return cv::imwrite((std::filesystem::path(dir) / name.str()).string(), img);
}

inline cv::Size dynamic_tuning_canvas_size(const WarpPackage& pack, const AppOptions& opt) {
    if (pack.valid && pack.warp_size.width > 0 && pack.warp_size.height > 0) return pack.warp_size;
    return cv::Size(std::max(1, opt.warp_width), std::max(1, opt.warp_height));
}

inline std::string dynamic_param_status(const DynamicRedRoiConfig& cfg, int step_px) {
    std::ostringstream oss;
    oss << "band=[" << cfg.band_y0 << ',' << cfg.band_y1 << ") h=" << (cfg.band_y1 - cfg.band_y0)
        << " gap=" << cfg.roi_gap_above_band
        << " roi=" << cfg.roi_width << 'x' << cfg.roi_height
        << " dual=" << (cfg.require_dual_zone ? "on" : "off")
        << " zone_gap=" << cfg.zone_gap_px
        << " zone_px>=" << cfg.zone_min_pixels
        << " zone_ratio>=" << std::fixed << std::setprecision(3) << cfg.zone_min_ratio;
    if (step_px > 0) oss << " step=" << step_px;
    return oss.str();
}

inline std::string red_status_text(const RoiRuntimeData& roi_info) {
    if (roi_info.runtime_mode == "fixed") {
        return "red_ratio=" + std::to_string(roi_info.red_ratio).substr(0, 6);
    }
    std::ostringstream oss;
    oss << "red_ratio=" << std::to_string(roi_info.red_ratio).substr(0, 6);
    if (roi_info.left_zone_found) oss << " L=" << roi_info.left_zone_center_x << "(" << roi_info.left_zone_red_pixels << "," << std::fixed << std::setprecision(3) << roi_info.left_zone_red_ratio << ")";
    else oss << " L=miss(" << roi_info.left_zone_red_pixels << "," << std::fixed << std::setprecision(3) << roi_info.left_zone_red_ratio << ")";
    if (roi_info.right_zone_found) oss << " R=" << roi_info.right_zone_center_x << "(" << roi_info.right_zone_red_pixels << "," << std::fixed << std::setprecision(3) << roi_info.right_zone_red_ratio << ")";
    else oss << " R=miss(" << roi_info.right_zone_red_pixels << "," << std::fixed << std::setprecision(3) << roi_info.right_zone_red_ratio << ")";
    if (roi_info.red_center_x >= 0) oss << " cx=" << roi_info.red_center_x;
    if (roi_info.dual_zone_triggered || roi_info.trigger_ready) oss << " trigger=ready";
    else oss << " trigger=wait";
    return oss.str();
}

inline std::vector<std::string> wrap_console_line(const std::string& line, int max_chars) {
    std::vector<std::string> out;
    if (max_chars <= 8 || static_cast<int>(line.size()) <= max_chars) {
        out.push_back(line);
        return out;
    }
    std::string remaining = line;
    while (!remaining.empty()) {
        if (static_cast<int>(remaining.size()) <= max_chars) {
            out.push_back(remaining);
            break;
        }
        int cut = max_chars;
        for (int i = max_chars; i > max_chars / 2; --i) {
            if (remaining[static_cast<size_t>(i)] == ' ') { cut = i; break; }
        }
        out.push_back(remaining.substr(0, static_cast<size_t>(cut)));
        size_t next = static_cast<size_t>(cut);
        while (next < remaining.size() && remaining[next] == ' ') ++next;
        remaining = remaining.substr(next);
    }
    return out;
}

inline void show_text_console(const std::string& window_name,
                              const std::vector<std::string>& lines,
                              int min_w = 560,
                              int min_h = 420) {
    int W = min_w;
    int H = min_h;
    try {
        const cv::Rect r = cv::getWindowImageRect(window_name);
        if (r.width > 64) W = r.width;
        if (r.height > 64) H = r.height;
    } catch (...) {}
    cv::Mat canvas(H, W, CV_8UC3, cv::Scalar(18, 18, 18));
    const int margin = 12;
    const int line_h = 20;
    const int max_chars = std::max(18, (W - 2 * margin) / 8);
    int y = margin + 18;
    for (const auto& line : lines) {
        for (const auto& wrapped : wrap_console_line(line, max_chars)) {
            if (y > H - margin) break;
            cv::putText(canvas, wrapped, {margin, y}, cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
            y += line_h;
        }
        if (y > H - margin) break;
    }
    cv::imshow(window_name, canvas);
}

inline std::vector<std::string> build_calibrate_console_lines(bool locked,
                                                              const std::string& roi_mode,
                                                              const DynamicRedRoiConfig& dynamic_cfg,
                                                              const RoiRuntimeData* runtime,
                                                              const WarpPackage& locked_pack,
                                                              int dynamic_step_px) {
    std::vector<std::string> lines;
    lines.push_back("calibrate | lock=" + std::string(locked ? "locked" : "search") + " | roi_mode=" + roi_mode);
    if (locked_pack.valid) {
        lines.push_back("warp=" + std::to_string(locked_pack.warp_size.width) + "x" + std::to_string(locked_pack.warp_size.height) +
                        " tag_px=" + std::to_string(locked_pack.target_tag_px) +
                        " id=" + std::to_string(locked_pack.id));
    }
    if (is_dynamic_roi_mode(roi_mode)) {
        lines.push_back(dynamic_param_status(dynamic_cfg, dynamic_step_px));
        if (runtime) lines.push_back(red_status_text(*runtime));
        lines.push_back("keys dyn: w/s band, i/k band_h, a/d roi_w, z/x roi_h, j/l gap, n/b zone_gap, c/v zone_px, f/g zone_ratio, [/] step");
    } else {
        lines.push_back("keys fixed: 1/2 select roi, wasd move, ijkl size, [ ] move-step, , . size-step");
    }
    lines.push_back("general: SPACE lock, u unlock, m mode, p save-all, y save-warp, o save-rois, r reset, q quit");
    return lines;
}

inline std::vector<std::string> build_deploy_console_lines(const std::string& roi_mode,
                                                           const DynamicRedRoiConfig& dynamic_cfg,
                                                           const RoiRuntimeData* runtime,
                                                           const std::string& cfg_warn,
                                                           int dynamic_step_px,
                                                           double warp_ms,
                                                           double roi_ms,
                                                           double model_ms) {
    std::vector<std::string> lines;
    lines.push_back("deploy | roi_mode=" + roi_mode);
    if (!cfg_warn.empty()) lines.push_back(cfg_warn);
    if (is_dynamic_roi_mode(roi_mode)) {
        lines.push_back(dynamic_param_status(dynamic_cfg, dynamic_step_px));
        if (runtime) lines.push_back(red_status_text(*runtime));
        lines.push_back("trigger rule: both left+right zones need valid blobs AND enough red portion before image ROI is emitted");
        lines.push_back("keys dyn: m mode, w/s band, i/k band_h, a/d roi_w, z/x roi_h, j/l gap, n/b zone_gap, c/v zone_px, f/g zone_ratio, [/] step");
    } else {
        if (runtime) lines.push_back(red_status_text(*runtime));
        lines.push_back("keys: m mode, q quit");
    }
    std::ostringstream perf;
    perf << "timing ms | warp=" << std::fixed << std::setprecision(3) << warp_ms
         << " roi=" << roi_ms << " model=" << model_ms;
    lines.push_back(perf.str());
    return lines;
}

inline void draw_runtime_roi_overlay(cv::Mat& img,
                                     const RoiConfig& rois,
                                     const DynamicRedRoiConfig& dynamic_cfg,
                                     const std::string& roi_mode,
                                     int selected,
                                     const RoiRuntimeData* runtime = nullptr) {
    if (!is_dynamic_roi_mode(roi_mode)) {
        draw_rois(img, rois, selected);
        return;
    }

    const cv::Rect search_rect = (runtime && runtime->dynamic_search_rect.area() > 0)
        ? runtime->dynamic_search_rect
        : dynamic_search_rect(dynamic_cfg, img.size());
    const int preview_center_x = search_rect.x + search_rect.width / 2;
    const cv::Rect image_rect = (runtime && runtime->dynamic_image_rect.area() > 0)
        ? runtime->dynamic_image_rect
        : dynamic_image_roi_rect(preview_center_x, dynamic_cfg, img.size());

    cv::rectangle(img, search_rect, cv::Scalar(0, 180, 255), 2);
    cv::putText(img, "red_band", search_rect.tl() + cv::Point(4, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 180, 255), 2);

    const cv::Rect left_zone = (runtime && runtime->dynamic_left_zone_rect.area() > 0)
        ? runtime->dynamic_left_zone_rect : dynamic_left_zone_rect(dynamic_cfg, img.size());
    const cv::Rect right_zone = (runtime && runtime->dynamic_right_zone_rect.area() > 0)
        ? runtime->dynamic_right_zone_rect : dynamic_right_zone_rect(dynamic_cfg, img.size());
    cv::rectangle(img, left_zone, cv::Scalar(80, 180, 255), 1);
    cv::rectangle(img, right_zone, cv::Scalar(80, 180, 255), 1);

    if (runtime && runtime->dynamic_image_rect.area() > 0) {
        cv::rectangle(img, image_rect, cv::Scalar(255, 0, 0), 2);
        cv::putText(img, "dynamic_image_roi", image_rect.tl() + cv::Point(4, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(255, 0, 0), 2);
    }

    const int center_x = (runtime && runtime->red_center_x >= 0) ? runtime->red_center_x : preview_center_x;
    cv::line(img, cv::Point(center_x, search_rect.y), cv::Point(center_x, search_rect.y + search_rect.height),
             cv::Scalar(0, 255, 0), 2);
    cv::putText(img, "x_center", cv::Point(std::max(0, center_x - 26), std::max(18, search_rect.y - 6)),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 180, 0), 1);

    if (runtime) {
        cv::putText(img, red_status_text(*runtime), {12, 76},
                    cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(0, 120, 0), 1);
    } else {
        cv::putText(img, dynamic_param_status(dynamic_cfg, 0), {12, 76},
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(0, 120, 0), 1);
    }
}

inline bool run_calibrate(const AppOptions& opt, std::string& err) {
    int cam_w = opt.width, cam_h = opt.height;
    cv::VideoCapture cap;
    if (!open_capture(cap, opt.device, cam_w, cam_h, opt.fps, opt.fourcc, opt.buffer_size,
                      opt.camera_soft_max, err)) return false;

    RoiConfig rois = opt.default_rois;
    DynamicRedRoiConfig dynamic_cfg = opt.dynamic_red;
    std::string roi_mode = normalize_roi_mode(opt.roi_mode);
    (void)load_rois_yaml(opt.load_rois, rois, &dynamic_cfg, &roi_mode);

    bool model_ready = false;
    if (opt.model_cfg.enable && opt.run_model) {
        std::string merr;
        model_ready = init_model_runtime(opt.model_cfg, merr);
        if (!model_ready) std::cerr << "\n[model] disabled: " << merr << "\n";
    }

    AprilTagConfig tag_cfg;
    tag_cfg.family = opt.tag_family;
    tag_cfg.target_id = opt.target_id;
    tag_cfg.require_target_id = opt.require_target_id;
    tag_cfg.manual_lock_only = opt.manual_lock_only;
    tag_cfg.lock_frames = opt.lock_frames;
    TagLocker locker(opt.lock_frames);

    bool locked = false;
    int selected = 0;
    double move_step = 0.01;
    double size_step = 0.01;
    int dynamic_step_px = 4;
    int frame_idx = 0;
    AprilTagDetection cur, locked_det;
    WarpPackage locked_pack;
    DynamicRedRoiState dynamic_state;
    ModelResult model_res;
    RoiRuntimeData roi_info;
    cv::Mat frame, camera_show, warp_show, temp_preview, warped, valid;

    if (opt.ui) {
        cv::namedWindow("vision_app_camera", cv::WINDOW_NORMAL);
        cv::namedWindow("vision_app_warp", cv::WINDOW_NORMAL);
        if (opt.text_console) cv::namedWindow("vision_app_text", cv::WINDOW_NORMAL);
        cv::resizeWindow("vision_app_camera", opt.camera_preview_max, std::max(240, opt.camera_preview_max * 3 / 4));
        cv::resizeWindow("vision_app_warp", opt.warp_preview_max, opt.warp_preview_max);
        if (opt.text_console) cv::resizeWindow("vision_app_text", 760, 420);
        if (dynamic_cfg.show_mask_window) cv::namedWindow("vision_app_red_mask", cv::WINDOW_NORMAL);
        print_calibrate_controls();
    }

    while (true) {
        if (!grab_latest_frame(cap, opt.latest_only, opt.drain_grabs, frame)) {
            err = "failed to read frame";
            break;
        }

        if (!locked) {
            std::string derr;
            detect_apriltag_best(frame, tag_cfg, cur, derr);
            const bool stable = locker.update(cur);
            if (!opt.manual_lock_only && stable && cur.found) {
                if (build_warp_package_from_detection(cur, frame.size(), opt.warp_width, opt.warp_height,
                                                     opt.target_tag_px, opt.warp_center_x_ratio, opt.warp_center_y_ratio, locked_pack, err)) {
                    locked = true;
                    locked_det = cur;
                    std::cout << "\n[lock] auto lock family=" << locked_det.family << " id=" << locked_det.id << "\n";
                }
                err.clear();
            }

            camera_show = frame.clone();
            if (opt.draw_overlay) {
                draw_detection_overlay(camera_show, cur);
                cv::putText(camera_show,
                            cur.found ? "SEARCH: centered live warp in other window" : "SEARCH: no tag",
                            {12, 56}, cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(0,255,255), 2);
            }
            camera_show = downscale_for_preview(camera_show, opt.camera_preview_max);

            temp_preview.release();
            if (cur.found) {
                std::string werr;
                WarpPackage temp_pack;
                if (build_warp_package_from_detection(cur, frame.size(), opt.warp_width, opt.warp_height,
                                                     opt.target_tag_px, opt.warp_center_x_ratio, opt.warp_center_y_ratio, temp_pack, werr)) {
                    cv::Mat temp_valid;
                    if (apply_warp(frame, temp_pack, temp_preview, &temp_valid)) {
                        if (opt.draw_overlay) {
                            draw_runtime_roi_overlay(temp_preview, rois, dynamic_cfg, roi_mode, selected, nullptr);
                            cv::putText(temp_preview,
                                        cv::format("SEARCH warp=%dx%d tag_px=%d mode=%s",
                                                   temp_pack.warp_size.width, temp_pack.warp_size.height,
                                                   temp_pack.target_tag_px, roi_mode.c_str()),
                                        {12, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(0, 100, 0), 2);
                        }
                    }
                }
            }
            if (temp_preview.empty()) {
                warp_show = make_blank_preview(opt.warp_width, opt.warp_height, opt.draw_overlay ? "waiting for tag" : "");
            } else {
                warp_show = temp_preview.clone();
            }
        } else {
            cur = locked_det;
            if (!apply_warp(frame, locked_pack, warped, &valid)) {
                err = "failed to apply locked warp";
                break;
            }
            camera_show = frame.clone();
            if (opt.draw_overlay) {
                draw_detection_overlay(camera_show, locked_det);
                cv::putText(camera_show, "LOCKED", {12, 56}, cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(0,255,0), 2);
            }

            warp_show = warped.clone();
            std::string rerr;
            if (extract_runtime_rois(warped, valid, roi_mode, rois, opt.red_cfg, dynamic_cfg,
                                     dynamic_state, roi_info, rerr)) {
                if (opt.draw_overlay) {
                    draw_runtime_roi_overlay(warp_show, rois, dynamic_cfg, roi_mode, selected, &roi_info);
                    cv::putText(warp_show,
                                cv::format("LOCKED family=%s id=%d warp=%dx%d tag_px=%d mode=%s",
                                           locked_pack.family.c_str(), locked_pack.id,
                                           locked_pack.warp_size.width, locked_pack.warp_size.height,
                                           locked_pack.target_tag_px, roi_mode.c_str()),
                                {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(0,120,0), 2);
                }
                if (!roi_info.trigger_ready) model_res = {};
                if (model_ready && opt.run_model && roi_info.trigger_ready) {
                    std::string merr;
                    if (!run_model_on_image_roi(roi_info, opt.model_cfg, model_res, merr)) {
                        model_res = {};
                        model_res.ran = true;
                        model_res.summary = "model err: " + merr;
                    }
                    if (opt.draw_overlay && model_res.ran) {
                        cv::putText(warp_show, model_res.summary, {12, 100},
                                    cv::FONT_HERSHEY_SIMPLEX, 0.46, cv::Scalar(0,120,0), 1);
                    }
                }
                if (opt.ui && dynamic_cfg.show_mask_window && !roi_info.red_mask_vis.empty()) {
                    cv::imshow("vision_app_red_mask", roi_info.red_mask_vis);
                }
            } else if (opt.draw_overlay && !rerr.empty()) {
                cv::putText(warp_show, rerr, {12, 56}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 2);
            }
        }

        if (opt.ui) {
            cv::imshow("vision_app_camera", downscale_for_preview(camera_show, opt.camera_preview_max));
            cv::imshow("vision_app_warp", downscale_for_preview(warp_show, opt.warp_preview_max));
            if (opt.text_console) {
                show_text_console("vision_app_text", build_calibrate_console_lines(locked, roi_mode, dynamic_cfg, locked ? &roi_info : nullptr, locked_pack, dynamic_step_px));
            }
        }

        ++frame_idx;
        int key = -1;
        if (opt.ui) key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') break;
        if (key == 'u') {
            locked = false;
            locker.reset();
            dynamic_state.reset();
            locked_pack = {};
            temp_preview.release();
            model_res = {};
            std::cout << "\n[lock] released\n";
        }
        if (key == 'm') {
            roi_mode = is_dynamic_roi_mode(roi_mode) ? std::string("fixed") : std::string("dynamic-red-x");
            dynamic_state.reset();
            model_res = {};
            std::cout << "\n[mode] roi_mode=" << roi_mode << "\n";
        }
        if (key == ' ' || key == 13) {
            if (cur.found) {
                if (build_warp_package_from_detection(cur, frame.size(), opt.warp_width, opt.warp_height,
                                                     opt.target_tag_px, opt.warp_center_x_ratio, opt.warp_center_y_ratio, locked_pack, err)) {
                    locked = true;
                    dynamic_state.reset();
                    locked_det = cur;
                    std::cout << "\n[lock] manual lock family=" << locked_det.family << " id=" << locked_det.id << "\n";
                } else {
                    std::cerr << "\n[lock] rejected: " << err << "\n";
                    err.clear();
                }
            }
        }
        if (key == '1') {
            selected = 0;
            std::cout << "\n[roi] selected red_roi" << (is_dynamic_roi_mode(roi_mode) ? " (fixed mode only)" : "") << "\n";
        }
        if (key == '2') {
            selected = 1;
            std::cout << "\n[roi] selected image_roi" << (is_dynamic_roi_mode(roi_mode) ? " (fixed mode only)" : "") << "\n";
        }
        if (!is_dynamic_roi_mode(roi_mode)) {
            if (key == '[') { move_step = std::max(0.001, move_step * 0.5); std::cout << "\n[step] move_step=" << move_step << "\n"; }
            if (key == ']') { move_step = std::min(0.25, move_step * 2.0); std::cout << "\n[step] move_step=" << move_step << "\n"; }
            if (key == ',') { size_step = std::max(0.001, size_step * 0.5); std::cout << "\n[step] size_step=" << size_step << "\n"; }
            if (key == '.') { size_step = std::min(0.25, size_step * 2.0); std::cout << "\n[step] size_step=" << size_step << "\n"; }
        } else {
            if (key == '[') { dynamic_step_px = std::max(1, dynamic_step_px / 2); std::cout << "\n[dyn] step_px=" << dynamic_step_px << "\n"; }
            if (key == ']') { dynamic_step_px = std::min(128, dynamic_step_px * 2); std::cout << "\n[dyn] step_px=" << dynamic_step_px << "\n"; }
        }
        if (key == 'h') { print_calibrate_controls(); }
        if (key == 'r') {
            rois = opt.default_rois;
            dynamic_cfg = opt.dynamic_red;
            dynamic_state.reset();
            std::cout << "\n[roi] reset to defaults\n";
        }
        if (locked && !is_dynamic_roi_mode(roi_mode)) {
            if (selected == 0) adjust_roi(rois.red_roi, key, move_step, size_step);
            else adjust_roi(rois.image_roi, key, move_step, size_step);
        } else if (is_dynamic_roi_mode(roi_mode)) {
            const cv::Size dyn_sz = dynamic_tuning_canvas_size(locked_pack, opt);
            const auto before = dynamic_param_status(dynamic_cfg, dynamic_step_px);
            tune_dynamic_red_cfg(dynamic_cfg, key, dynamic_step_px, dyn_sz);
            if (before != dynamic_param_status(dynamic_cfg, dynamic_step_px)) {
                dynamic_state.reset();
                std::cout << "\n[dyn] " << dynamic_param_status(dynamic_cfg, dynamic_step_px) << "\n";
            }
        }
        if (key == 'o') {
            if (save_rois_yaml(opt.save_rois, rois, &dynamic_cfg, &roi_mode)) {
                std::cout << "\n[save] rois -> " << opt.save_rois << "\n";
            }
        }
        if (key == 'y' && locked) {
            if (save_warp_package(opt.save_warp, locked_pack)) std::cout << "\n[save] warp -> " << opt.save_warp << "\n";
        }
        if (key == 'p' && locked) {
            save_rois_yaml(opt.save_rois, rois, &dynamic_cfg, &roi_mode);
            save_warp_package(opt.save_warp, locked_pack);
            std::ostringstream extra;
            extra << "- Locked family: " << locked_pack.family << "\n"
                  << "- Locked id: " << locked_pack.id << "\n"
                  << "- Source size: " << locked_pack.src_size.width << 'x' << locked_pack.src_size.height << "\n"
                  << "- Warp size: " << locked_pack.warp_size.width << 'x' << locked_pack.warp_size.height << "\n"
                  << "- Target tag px: " << locked_pack.target_tag_px << "\n"
                  << "- roi_mode: " << roi_mode << "\n"
                  << "- red_roi ratio: " << rois.red_roi.x << ',' << rois.red_roi.y << ',' << rois.red_roi.w << ',' << rois.red_roi.h << "\n"
                  << "- image_roi ratio: " << rois.image_roi.x << ',' << rois.image_roi.y << ',' << rois.image_roi.w << ',' << rois.image_roi.h << "\n"
                  << "- dynamic band: y0=" << dynamic_cfg.band_y0 << " y1=" << dynamic_cfg.band_y1
                  << " x0=" << dynamic_cfg.search_x0 << " x1=" << dynamic_cfg.search_x1 << "\n"
                  << "- dynamic crop: gap_above_band=" << dynamic_cfg.roi_gap_above_band
                  << " anchor_y_legacy=" << dynamic_cfg.roi_anchor_y
                  << " w=" << dynamic_cfg.roi_width
                  << " h=" << dynamic_cfg.roi_height << "\n"
                  << "- dynamic blob: min_area=" << dynamic_cfg.min_area << " max_area=" << dynamic_cfg.max_area
                  << " morph_k=" << dynamic_cfg.morph_k << " alpha=" << dynamic_cfg.center_alpha
                  << " miss_tolerance=" << dynamic_cfg.miss_tolerance;
            write_report_md(opt.save_report, "Calibration Report", nullptr, extra.str());
            std::cout << "\n[save] all -> " << opt.save_warp << " ; " << opt.save_rois << " ; " << opt.save_report << "\n";
        }
    }

    release_model_runtime();
    if (opt.ui) cv::destroyAllWindows();
    std::cout << "\n";
    return err.empty();
}

inline bool run_deploy(const AppOptions& opt, std::string& err) {
    WarpPackage pack;
    if (!load_warp_package(opt.load_warp, pack)) {
        err = "failed to load warp package: " + opt.load_warp;
        return false;
    }
    RoiConfig rois = opt.default_rois;
    DynamicRedRoiConfig dynamic_cfg = opt.dynamic_red;
    std::string roi_mode = normalize_roi_mode(opt.roi_mode);
    (void)load_rois_yaml(opt.load_rois, rois, &dynamic_cfg, &roi_mode);
    DynamicRedRoiState dynamic_state;

    bool model_ready = false;
    if (opt.model_cfg.enable && opt.run_model) {
        std::string merr;
        model_ready = init_model_runtime(opt.model_cfg, merr);
        if (!model_ready) std::cerr << "\n[model] disabled: " << merr << "\n";
    }

    int cam_w = opt.width, cam_h = opt.height;
    cv::VideoCapture cap;
    if (!open_capture(cap, opt.device, cam_w, cam_h, opt.fps, opt.fourcc, opt.buffer_size,
                      opt.camera_soft_max, err)) return false;

    const std::string cfg_warn = build_deploy_config_warning(opt, pack, cam_w, cam_h);
    if (!cfg_warn.empty()) std::cerr << cfg_warn;

    cv::Mat frame, warped, valid, camera_show, warp_show;
    RoiRuntimeData roi_info;
    ModelResult model_res;
    if (opt.ui) {
        cv::namedWindow("vision_app_camera", cv::WINDOW_NORMAL);
        cv::namedWindow("vision_app_warp", cv::WINDOW_NORMAL);
        if (opt.text_console) cv::namedWindow("vision_app_text", cv::WINDOW_NORMAL);
        cv::resizeWindow("vision_app_camera", opt.camera_preview_max, std::max(240, opt.camera_preview_max * 3 / 4));
        cv::resizeWindow("vision_app_warp", opt.warp_preview_max, opt.warp_preview_max);
        if (opt.text_console) cv::resizeWindow("vision_app_text", 760, 420);
        if (dynamic_cfg.show_mask_window) cv::namedWindow("vision_app_red_mask", cv::WINDOW_NORMAL);
    }
    std::cout << "\n=== vision_app deploy ===\n"
              << "loaded warp: " << opt.load_warp << "\n"
              << "loaded rois: " << opt.load_rois << "\n"
              << "roi_mode: " << roi_mode << "\n"
              << "backend: " << opt.model_cfg.backend << "\n"
              << "run_red=" << opt.run_red << " run_image_roi=" << opt.run_image_roi << " run_model=" << opt.run_model << "\n"
              << "keys: m toggle mode | dynamic => w/s band, i/k band h, a/d roi w, z/x roi h, j/l gap, [/] step\n";

    using Clock = std::chrono::steady_clock;
    Clock::time_point last_model_tp{};
    bool have_last_model_tp = false;
    int dynamic_step_px = 4;
    int frame_idx = 0;
    auto loop_t0 = Clock::now();

    while (true) {
        auto frame_t0 = Clock::now();
        if (!grab_latest_frame(cap, opt.latest_only, opt.drain_grabs, frame)) { err = "failed to read frame"; break; }
        auto warp_t0 = Clock::now();
        if (!apply_warp(frame, pack, warped, &valid)) { err = "failed to apply warp"; break; }
        const double warp_ms = std::chrono::duration<double, std::milli>(Clock::now() - warp_t0).count();

        camera_show = frame.clone();
        warp_show = warped.clone();

        std::string rerr;
        double roi_ms = 0.0;
        auto roi_t0 = Clock::now();
        if (extract_runtime_rois(warped, valid, roi_mode, rois, opt.red_cfg, dynamic_cfg,
                                 dynamic_state, roi_info, rerr)) {
            roi_ms = std::chrono::duration<double, std::milli>(Clock::now() - roi_t0).count();
            if (opt.draw_overlay) {
                draw_runtime_roi_overlay(warp_show, rois, dynamic_cfg, roi_mode, -1, &roi_info);
                cv::putText(warp_show, cv::format("DEPLOY family=%s id=%d mode=%s",
                                                  pack.family.c_str(), pack.id, roi_mode.c_str()),
                            {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.68, cv::Scalar(0,120,0), 2);
                if (!cfg_warn.empty()) {
                    cv::putText(warp_show, "WARN: camera/config mismatch - see terminal", {12, 52},
                                cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(0,0,255), 1);
                }
            }

            if (opt.save_every_n > 0 && frame_idx % opt.save_every_n == 0) {
                if (opt.run_image_roi) save_crop_if_needed(opt.save_image_roi_dir, "image_roi", frame_idx, roi_info.image_bgr);
                if (opt.run_red) save_crop_if_needed(opt.save_red_roi_dir, is_dynamic_roi_mode(roi_mode) ? "red_band" : "red_roi",
                                                     frame_idx, roi_info.red_bgr);
            }

            const bool stride_ok = (opt.model_cfg.stride <= 1) || ((frame_idx % opt.model_cfg.stride) == 0);
            bool hz_ok = true;
            if (opt.model_max_hz > 0.0 && have_last_model_tp) {
                const double dt = std::chrono::duration<double>(Clock::now() - last_model_tp).count();
                hz_ok = dt >= (1.0 / opt.model_max_hz);
            }

            if (!roi_info.trigger_ready) model_res = {};
            if (model_ready && opt.run_model && opt.run_image_roi && roi_info.trigger_ready && stride_ok && hz_ok) {
                std::string merr;
                if (!run_model_on_image_roi(roi_info, opt.model_cfg, model_res, merr)) {
                    model_res = {};
                    model_res.ran = true;
                    model_res.summary = "model err: " + merr;
                }
                last_model_tp = Clock::now();
                have_last_model_tp = true;
            }

            if (opt.draw_overlay && opt.run_model && model_res.ran) {
                cv::putText(warp_show, model_res.summary, {12, 100},
                            cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(0,120,0), 1);
            }

            if (opt.draw_overlay) {
                const double total_ms = std::chrono::duration<double, std::milli>(Clock::now() - frame_t0).count();
                const double fps = (total_ms > 0.0) ? (1000.0 / total_ms) : 0.0;
                cv::putText(warp_show,
                            cv::format("fps=%.2f warp=%.3f roi=%.3f model=%.3f", fps, warp_ms, roi_ms, model_res.infer_ms),
                            {12, 122}, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0,120,0), 1);
                cv::putText(warp_show,
                            cv::format("model_hz<=%.2f stride=%d", opt.model_max_hz, opt.model_cfg.stride),
                            {12, 142}, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0,120,0), 1);
                if (is_dynamic_roi_mode(roi_mode)) {
                    cv::putText(warp_show, dynamic_param_status(dynamic_cfg, dynamic_step_px),
                                {12, 162}, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0,120,0), 1);
                }
            }

            if (opt.ui && dynamic_cfg.show_mask_window && !roi_info.red_mask_vis.empty()) {
                cv::imshow("vision_app_red_mask", roi_info.red_mask_vis);
            }
        } else if (opt.draw_overlay && !rerr.empty()) {
            cv::putText(warp_show, rerr, {12, 76}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 2);
        }

        if (opt.ui) {
            cv::imshow("vision_app_camera", downscale_for_preview(camera_show, opt.camera_preview_max));
            cv::imshow("vision_app_warp", downscale_for_preview(warp_show, opt.warp_preview_max));
            if (opt.text_console) {
                show_text_console("vision_app_text", build_deploy_console_lines(roi_mode, dynamic_cfg, &roi_info, cfg_warn, dynamic_step_px, warp_ms, roi_ms, model_res.infer_ms));
            }
        }

        ++frame_idx;
        int key = -1;
        if (opt.ui) key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') break;
        if (key == 'm') {
            roi_mode = is_dynamic_roi_mode(roi_mode) ? std::string("fixed") : std::string("dynamic-red-x");
            dynamic_state.reset();
            model_res = {};
            std::cout << "\n[mode] roi_mode=" << roi_mode << "\n";
        }
        if (is_dynamic_roi_mode(roi_mode)) {
            if (key == '[') { dynamic_step_px = std::max(1, dynamic_step_px / 2); std::cout << "\n[dyn] step_px=" << dynamic_step_px << "\n"; }
            if (key == ']') { dynamic_step_px = std::min(128, dynamic_step_px * 2); std::cout << "\n[dyn] step_px=" << dynamic_step_px << "\n"; }
            const cv::Size dyn_sz = dynamic_tuning_canvas_size(pack, opt);
            const auto before = dynamic_param_status(dynamic_cfg, dynamic_step_px);
            tune_dynamic_red_cfg(dynamic_cfg, key, dynamic_step_px, dyn_sz);
            if (before != dynamic_param_status(dynamic_cfg, dynamic_step_px)) {
                dynamic_state.reset();
                model_res = {};
                std::cout << "\n[dyn] " << dynamic_param_status(dynamic_cfg, dynamic_step_px) << "\n";
            }
        }

        if (!opt.ui && opt.duration > 0) {
            const double elapsed = std::chrono::duration<double>(Clock::now() - loop_t0).count();
            if (elapsed >= opt.duration) break;
        }
    }

    release_model_runtime();
    if (opt.ui) cv::destroyAllWindows();
    std::cout << "\n";
    return err.empty();
}

} // namespace vision_app
