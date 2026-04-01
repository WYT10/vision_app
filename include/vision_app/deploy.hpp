#pragma once

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "camera.hpp"
#include "calibrate.hpp"
#include "model.hpp"
#include "stats.hpp"
#include "text_console.hpp"

namespace vision_app {

struct RedThresholdConfig {
    int h1_low = 0;
    int h1_high = 10;
    int h2_low = 170;
    int h2_high = 180;
    int s_min = 80;
    int v_min = 60;
    int morph_k = 3;
};

struct DynamicRedStackedConfig {
    int search_x0 = 0;
    int search_x1 = -1; // -1 = full width

    int upper_y0 = 110;
    int upper_y1 = 135;
    int lower_y0 = 145;
    int lower_y1 = 175;

    int zone_min_pixels = 24;
    double zone_min_ratio = 0.015;
    int center_x_max_diff = 28;
    int stable_frames_required = 2;

    int roi_width = 96;
    int roi_height = 96;
    int roi_gap_above_upper_zone = 0;
};

inline void clamp_dynamic_cfg(DynamicRedStackedConfig& d, const cv::Size& sz) {
    const int W = std::max(1, sz.width);
    const int H = std::max(1, sz.height);
    d.search_x0 = std::clamp(d.search_x0, 0, W - 1);
    if (d.search_x1 < 0) d.search_x1 = W;
    d.search_x1 = std::clamp(d.search_x1, d.search_x0 + 1, W);
    d.upper_y0 = std::clamp(d.upper_y0, 0, H - 1);
    d.upper_y1 = std::clamp(d.upper_y1, d.upper_y0 + 1, H);
    d.lower_y0 = std::clamp(d.lower_y0, d.upper_y1, H - 1);
    d.lower_y1 = std::clamp(d.lower_y1, d.lower_y0 + 1, H);
    d.zone_min_pixels = std::max(1, d.zone_min_pixels);
    d.zone_min_ratio = std::clamp(d.zone_min_ratio, 0.0, 1.0);
    d.center_x_max_diff = std::max(0, d.center_x_max_diff);
    d.stable_frames_required = std::max(1, d.stable_frames_required);
    d.roi_width = std::clamp(d.roi_width, 8, W);
    d.roi_height = std::clamp(d.roi_height, 8, H);
    d.roi_gap_above_upper_zone = std::max(0, d.roi_gap_above_upper_zone);
}

struct AppOptions {
    std::string mode = "probe";             // probe | calibrate | deploy | live
    std::string probe_task = "list";        // list | live | snap | bench
    std::string roi_mode = "fixed";         // fixed | dynamic_red_stacked

    std::string device = "/dev/video0";
    int width = 160;
    int height = 120;
    int fps = 120;
    std::string fourcc = "MJPG";
    int buffer_size = 1;
    bool latest_only = true;
    int drain_grabs = 1;
    bool ui = true;
    bool draw_overlay = true;
    bool text_console = true;
    bool red_show_mask_window = false;
    int duration = 10;
    std::string snap_path = "../report/snap.jpg";

    int camera_soft_max = 1000;
    int camera_preview_max = 640;
    int warp_preview_max = 640;

    int warp_width = 384;
    int warp_height = 384;
    int target_tag_px = 128;
    double warp_center_x_ratio = 0.50;
    double warp_center_y_ratio = 0.42;

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

    RoiConfig default_rois;
    RedThresholdConfig red_cfg;
    DynamicRedStackedConfig dyn_cfg;
    ModelConfig model_cfg;
    double model_max_hz = 5.0;

    bool run_red = true;
    bool run_image_roi = true;
    bool run_model = false;

    std::string save_image_roi_dir;
    std::string save_red_roi_dir;
    int save_every_n = 0;
};

bool extract_runtime_rois_fixed(const cv::Mat& warped,
                                const cv::Mat& valid_mask,
                                const RoiConfig& rois,
                                const RedThresholdConfig& red_cfg,
                                RoiRuntimeData& out,
                                std::string& err);

bool extract_runtime_rois_dynamic_stacked(const cv::Mat& warped,
                                          const cv::Mat& valid_mask,
                                          const DynamicRedStackedConfig& dyn_cfg,
                                          const RedThresholdConfig& red_cfg,
                                          int stable_counter,
                                          RoiRuntimeData& out,
                                          std::string& err);

inline std::string build_effective_config(const AppOptions& opt) {
    std::ostringstream oss;
    oss << "=== effective config ===\n";
    oss << "mode=" << opt.mode << "\n";
    oss << "probe_task=" << opt.probe_task << "\n";
    oss << "roi_mode=" << opt.roi_mode << "\n";
    oss << "camera=" << opt.device << " " << opt.width << "x" << opt.height
        << " fps=" << opt.fps << " fourcc=" << opt.fourcc << "\n";
    oss << "warp=" << opt.warp_width << "x" << opt.warp_height << " tag_px=" << opt.target_tag_px
        << " center=(" << opt.warp_center_x_ratio << "," << opt.warp_center_y_ratio << ")\n";
    oss << "upper_zone=[x:" << opt.dyn_cfg.search_x0 << ":" << opt.dyn_cfg.search_x1
        << ", y:" << opt.dyn_cfg.upper_y0 << ":" << opt.dyn_cfg.upper_y1 << "]\n";
    oss << "lower_zone=[x:" << opt.dyn_cfg.search_x0 << ":" << opt.dyn_cfg.search_x1
        << ", y:" << opt.dyn_cfg.lower_y0 << ":" << opt.dyn_cfg.lower_y1 << "]\n";
    oss << "dynamic_roi=(w=" << opt.dyn_cfg.roi_width << ", h=" << opt.dyn_cfg.roi_height
        << ", gap=" << opt.dyn_cfg.roi_gap_above_upper_zone << ")\n";
    return oss.str();
}

inline bool save_crop_if_needed(const std::string& dir,
                                const std::string& prefix,
                                int frame_idx,
                                const cv::Mat& img) {
    if (dir.empty() || img.empty()) return false;
    std::filesystem::create_directories(dir);
    std::ostringstream name;
    name << prefix << "_" << std::setw(6) << std::setfill('0') << frame_idx << ".jpg";
    return cv::imwrite((std::filesystem::path(dir) / name.str()).string(), img);
}

inline std::vector<std::string> build_dynamic_console_lines(const AppOptions& opt,
                                                            const RoiRuntimeData& roi_info,
                                                            int stable_counter) {
    std::vector<std::string> lines;
    lines.push_back("mode=" + opt.mode + " roi_mode=" + opt.roi_mode);
    lines.push_back(cv::format("upper zone x=[%d,%d) y=[%d,%d)", opt.dyn_cfg.search_x0, opt.dyn_cfg.search_x1,
                               opt.dyn_cfg.upper_y0, opt.dyn_cfg.upper_y1));
    lines.push_back(cv::format("lower zone x=[%d,%d) y=[%d,%d)", opt.dyn_cfg.search_x0, opt.dyn_cfg.search_x1,
                               opt.dyn_cfg.lower_y0, opt.dyn_cfg.lower_y1));
    lines.push_back(cv::format("upper valid=%d pixels=%d ratio=%.4f x=%.2f", roi_info.upper_valid ? 1 : 0,
                               roi_info.upper_red_pixels, roi_info.upper_red_ratio, roi_info.x_upper));
    lines.push_back(cv::format("lower valid=%d pixels=%d ratio=%.4f x=%.2f", roi_info.lower_valid ? 1 : 0,
                               roi_info.lower_red_pixels, roi_info.lower_red_ratio, roi_info.x_lower));
    lines.push_back(cv::format("trigger=%d stable=%d/%d center=%.2f max_dx=%d", roi_info.trigger_ready ? 1 : 0,
                               stable_counter, opt.dyn_cfg.stable_frames_required, roi_info.x_center,
                               opt.dyn_cfg.center_x_max_diff));
    lines.push_back(cv::format("roi x=%d y=%d w=%d h=%d", roi_info.image_roi_rect.x, roi_info.image_roi_rect.y,
                               roi_info.image_roi_rect.width, roi_info.image_roi_rect.height));
    lines.push_back(cv::format("tune: w/s move, t/g upper h, y/6 lower h, i/k gap, a/d roi_w, z/x roi_h, j/l roi_gap, c/v min_px, f/b min_ratio, n/m max_dx, [/] step, TAB mode"));
    return lines;
}

inline void draw_dynamic_overlay(cv::Mat& img, const RoiRuntimeData& roi) {
    cv::rectangle(img, roi.upper_zone_rect, cv::Scalar(0, 165, 255), 2);
    cv::putText(img, "upper_zone", roi.upper_zone_rect.tl() + cv::Point(4, 18),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 165, 255), 1, cv::LINE_AA);
    cv::rectangle(img, roi.lower_zone_rect, cv::Scalar(0, 215, 215), 2);
    cv::putText(img, "lower_zone", roi.lower_zone_rect.tl() + cv::Point(4, 18),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 215, 215), 1, cv::LINE_AA);

    if (roi.x_upper >= 0.0) {
        cv::line(img, {cvRound(roi.x_upper), roi.upper_zone_rect.y},
                 {cvRound(roi.x_upper), roi.upper_zone_rect.y + roi.upper_zone_rect.height},
                 cv::Scalar(60, 200, 255), 2);
    }
    if (roi.x_lower >= 0.0) {
        cv::line(img, {cvRound(roi.x_lower), roi.lower_zone_rect.y},
                 {cvRound(roi.x_lower), roi.lower_zone_rect.y + roi.lower_zone_rect.height},
                 cv::Scalar(60, 255, 200), 2);
    }
    if (roi.x_center >= 0.0) {
        cv::line(img, {cvRound(roi.x_center), 0}, {cvRound(roi.x_center), img.rows - 1},
                 roi.trigger_ready ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
    }
    if (roi.image_roi_rect.area() > 0) {
        cv::rectangle(img, roi.image_roi_rect, cv::Scalar(255, 0, 0), 2);
        cv::putText(img, "dynamic_image_roi", roi.image_roi_rect.tl() + cv::Point(4, 18),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
    }
}

inline std::string build_deploy_config_warning(const AppOptions& opt,
                                               const WarpPackage& pack,
                                               int cam_w,
                                               int cam_h) {
    std::ostringstream oss;
    bool any = false;
    if (pack.src_size.width > 0 && pack.src_size.height > 0 &&
        (cam_w != pack.src_size.width || cam_h != pack.src_size.height)) {
        oss << "[warn] live camera size " << cam_w << "x" << cam_h
            << " does not match saved warp source size "
            << pack.src_size.width << "x" << pack.src_size.height << "\n";
        any = true;
    }
    if ((opt.warp_width > 0 && opt.warp_width != pack.warp_size.width) ||
        (opt.warp_height > 0 && opt.warp_height != pack.warp_size.height) ||
        (opt.target_tag_px > 0 && pack.target_tag_px > 0 && opt.target_tag_px != pack.target_tag_px)) {
        oss << "[warn] saved warp size/tag_px differ from current CLI hints\n";
        any = true;
    }
    return any ? oss.str() : std::string();
}

inline void print_calibrate_controls() {
    std::cout << R"TXT(
=== vision_app calibrate controls ===
General
  SPACE / ENTER : lock current tag
  u             : unlock / reacquire
  TAB           : toggle fixed / dynamic_red_stacked
  p             : save all
  y             : save warp only
  o             : save rois only
  r             : reset tuned params
  h             : show help again
  q / ESC       : quit

Fixed mode
  1 / 2         : select red_roi / image_roi
  w a s d       : move ROI
  i / k         : height - / +
  j / l         : width  - / +
  [ / ]         : move step down / up
  , / .         : size step down / up

Dynamic stacked mode
  w / s         : move stacked zones up / down
  t / g         : upper zone height + / -
  y / 6         : lower zone height + / -
  i / k         : vertical gap between zones - / +
  a / d         : roi width - / +
  z / x         : roi height - / +
  j / l         : roi gap above upper zone - / +
  c / v         : zone min pixels - / +
  f / b         : zone min ratio - / +
  n / m         : center x max diff - / +
  [ / ]         : pixel step down / up
)TXT" << std::flush;
}

inline void tune_dynamic_cfg(DynamicRedStackedConfig& d, int key, int step) {
    if (key == 'w') { d.upper_y0 -= step; d.upper_y1 -= step; d.lower_y0 -= step; d.lower_y1 -= step; }
    if (key == 's') { d.upper_y0 += step; d.upper_y1 += step; d.lower_y0 += step; d.lower_y1 += step; }
    if (key == 't') d.upper_y1 += step;
    if (key == 'g') d.upper_y1 -= step;
    if (key == 'y') d.lower_y1 += step;
    if (key == '6') d.lower_y1 -= step;
    if (key == 'i') { d.lower_y0 -= step; d.lower_y1 -= step; }
    if (key == 'k') { d.lower_y0 += step; d.lower_y1 += step; }
    if (key == 'a') d.roi_width -= step;
    if (key == 'd') d.roi_width += step;
    if (key == 'z') d.roi_height -= step;
    if (key == 'x') d.roi_height += step;
    if (key == 'j') d.roi_gap_above_upper_zone = std::max(0, d.roi_gap_above_upper_zone - step);
    if (key == 'l') d.roi_gap_above_upper_zone += step;
    if (key == 'c') d.zone_min_pixels = std::max(1, d.zone_min_pixels - step);
    if (key == 'v') d.zone_min_pixels += step;
    if (key == 'f') d.zone_min_ratio = std::max(0.0, d.zone_min_ratio - 0.001 * step);
    if (key == 'b') d.zone_min_ratio = std::min(1.0, d.zone_min_ratio + 0.001 * step);
    if (key == 'n') d.center_x_max_diff = std::max(0, d.center_x_max_diff - step);
    if (key == 'm') d.center_x_max_diff += step;
}

inline bool run_probe(const AppOptions& opt, std::string& err) {
    err.clear();
    if (opt.probe_task == "list") {
        CameraProbeResult probe;
        if (!probe_camera(opt.device, probe, err)) return false;
        print_probe(probe);
        return true;
    }
    if (opt.probe_task == "bench") {
        RuntimeStats s;
        if (!bench_capture(opt.device, opt.width, opt.height, opt.fps, opt.fourcc,
                           opt.buffer_size, opt.latest_only, opt.drain_grabs,
                           !opt.ui, opt.duration, opt.camera_soft_max, opt.camera_preview_max,
                           s, err)) return false;
        print_runtime_stats(s);
        write_report_md(opt.save_report, "Probe Bench Report", &s, "- probe_task: bench");
        return true;
    }

    cv::VideoCapture cap;
    int w = opt.width, h = opt.height;
    if (!open_capture(cap, opt.device, w, h, opt.fps, opt.fourcc, opt.buffer_size, opt.camera_soft_max, err)) return false;
    cv::Mat frame;
    if (!grab_latest_frame(cap, opt.latest_only, opt.drain_grabs, frame)) {
        err = "failed to read frame";
        return false;
    }
    if (opt.probe_task == "snap") {
        std::filesystem::create_directories(std::filesystem::path(opt.snap_path).parent_path());
        if (!cv::imwrite(opt.snap_path, frame)) {
            err = "failed to save snap: " + opt.snap_path;
            return false;
        }
        std::cout << "saved snap -> " << opt.snap_path << "\n";
        return true;
    }
    if (opt.probe_task == "live") {
        if (!opt.ui) return true;
        cv::namedWindow("vision_app_probe", cv::WINDOW_NORMAL);
        const auto t0 = std::chrono::steady_clock::now();
        while (true) {
            if (!grab_latest_frame(cap, opt.latest_only, opt.drain_grabs, frame)) { err = "failed to read frame"; break; }
            cv::Mat show = downscale_for_preview(frame, opt.camera_preview_max);
            cv::putText(show, cv::format("probe live %dx%d", frame.cols, frame.rows), {12, 24},
                        cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0,255,0), 2);
            cv::imshow("vision_app_probe", show);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q') break;
            if (opt.duration > 0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() >= opt.duration) break;
        }
        cv::destroyWindow("vision_app_probe");
        return err.empty();
    }
    err = "unknown probe task: " + opt.probe_task;
    return false;
}

inline bool run_calibrate(const AppOptions& opt_in, std::string& err) {
    AppOptions opt = opt_in;
    int cam_w = opt.width, cam_h = opt.height;
    cv::VideoCapture cap;
    if (!open_capture(cap, opt.device, cam_w, cam_h, opt.fps, opt.fourcc, opt.buffer_size, opt.camera_soft_max, err)) return false;

    RoiConfig rois = opt.default_rois;
    (void)load_rois_yaml(opt.load_rois, rois);
    DynamicRedStackedConfig dyn = opt.dyn_cfg;
    clamp_dynamic_cfg(dyn, {opt.warp_width, opt.warp_height});

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
    int pixel_step = 4;
    int stable_counter = 0;
    int frame_idx = 0;
    AprilTagDetection cur, locked_det;
    WarpPackage locked_pack;
    RoiRuntimeData roi_info;
    cv::Mat frame, camera_show, warp_show, temp_preview, warped, valid;
    TextConsole console;
    console.enabled = opt.text_console && opt.ui;

    if (opt.ui) {
        cv::namedWindow("vision_app_camera", cv::WINDOW_NORMAL);
        cv::namedWindow("vision_app_warp", cv::WINDOW_NORMAL);
        cv::resizeWindow("vision_app_camera", opt.camera_preview_max, std::max(240, opt.camera_preview_max * 3 / 4));
        cv::resizeWindow("vision_app_warp", opt.warp_preview_max, opt.warp_preview_max);
        if (opt.red_show_mask_window) cv::namedWindow("vision_app_red_mask", cv::WINDOW_NORMAL);
        console.open();
        print_calibrate_controls();
    }

    while (true) {
        if (!grab_latest_frame(cap, opt.latest_only, opt.drain_grabs, frame)) { err = "failed to read frame"; break; }

        if (!locked) {
            std::string derr;
            detect_apriltag_best(frame, tag_cfg, cur, derr);
            const bool stable = locker.update(cur);
            if (!opt.manual_lock_only && stable && cur.found) {
                if (build_warp_package_from_detection(cur, frame.size(), opt.warp_width, opt.warp_height,
                                                     opt.target_tag_px, opt.warp_center_x_ratio, opt.warp_center_y_ratio,
                                                     locked_pack, err)) {
                    locked = true;
                    locked_det = cur;
                }
                err.clear();
            }
            camera_show = frame.clone();
            if (opt.draw_overlay) {
                draw_detection_overlay(camera_show, cur);
                cv::putText(camera_show, cur.found ? "SEARCH: tag found" : "SEARCH: no tag",
                            {12, 56}, cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(0,255,255), 2);
            }
            if (cur.found) {
                WarpPackage temp_pack;
                std::string werr;
                if (build_warp_package_from_detection(cur, frame.size(), opt.warp_width, opt.warp_height,
                                                     opt.target_tag_px, opt.warp_center_x_ratio, opt.warp_center_y_ratio,
                                                     temp_pack, werr) && apply_warp(frame, temp_pack, temp_preview, &valid)) {
                    warp_show = temp_preview.clone();
                    if (opt.roi_mode == "fixed") draw_rois(warp_show, rois, selected);
                    else {
                        std::string rerr;
                        extract_runtime_rois_dynamic_stacked(temp_preview, valid, dyn, opt.red_cfg, stable_counter, roi_info, rerr);
                        draw_dynamic_overlay(warp_show, roi_info);
                        if (opt.red_show_mask_window && !roi_info.red_mask.empty()) cv::imshow("vision_app_red_mask", roi_info.red_mask);
                    }
                }
            }
            if (warp_show.empty()) warp_show = cv::Mat(opt.warp_height, opt.warp_width, CV_8UC3, cv::Scalar(235,235,235));
            cv::putText(warp_show, "SEARCH / UNLOCKED", {12, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0,120,0), 2);
        } else {
            cur = locked_det;
            if (!apply_warp(frame, locked_pack, warped, &valid)) { err = "failed to apply locked warp"; break; }
            camera_show = frame.clone();
            if (opt.draw_overlay) {
                draw_detection_overlay(camera_show, locked_det);
                cv::putText(camera_show, "LOCKED", {12, 56}, cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(0,255,0), 2);
            }
            warp_show = warped.clone();
            std::string rerr;
            if (opt.roi_mode == "fixed") {
                if (extract_runtime_rois_fixed(warped, valid, rois, opt.red_cfg, roi_info, rerr)) draw_rois(warp_show, rois, selected);
            } else {
                if (extract_runtime_rois_dynamic_stacked(warped, valid, dyn, opt.red_cfg, stable_counter, roi_info, rerr)) {
                    stable_counter = roi_info.trigger_ready ? std::min(stable_counter + 1, dyn.stable_frames_required) : 0;
                    draw_dynamic_overlay(warp_show, roi_info);
                    if (opt.red_show_mask_window && !roi_info.red_mask.empty()) cv::imshow("vision_app_red_mask", roi_info.red_mask);
                }
            }
            if (!rerr.empty()) cv::putText(warp_show, rerr, {12, warp_show.rows - 12}, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0,0,255), 1);
        }

        if (opt.ui) {
            cv::imshow("vision_app_camera", downscale_for_preview(camera_show, opt.camera_preview_max));
            cv::imshow("vision_app_warp", downscale_for_preview(warp_show, opt.warp_preview_max));
            if (console.enabled) {
                if (opt.roi_mode == "dynamic_red_stacked") console.show(build_dynamic_console_lines(opt, roi_info, stable_counter));
                else console.show({"mode=calibrate roi_mode=fixed", "Fixed ROI tuning active.", "Keys: 1/2 select, WASD move, I/K height, J/L width, [/] move step, ,/. size step"});
            }
        }

        ++frame_idx;
        int key = opt.ui ? (cv::waitKey(1) & 0xFF) : -1;
        if (key == 27 || key == 'q') break;
        if (key == 'u') { locked = false; locker.reset(); stable_counter = 0; }
        if (key == 9) { opt.roi_mode = (opt.roi_mode == "fixed") ? "dynamic_red_stacked" : "fixed"; }
        if (key == '[') pixel_step = std::max(1, pixel_step / 2);
        if (key == ']') pixel_step = std::min(64, pixel_step * 2);
        if (key == 'h') print_calibrate_controls();
        if (key == 'r') { rois = opt.default_rois; dyn = opt.dyn_cfg; clamp_dynamic_cfg(dyn, {opt.warp_width, opt.warp_height}); }
        if (key == ' ' || key == 13) {
            if (cur.found) {
                if (build_warp_package_from_detection(cur, frame.size(), opt.warp_width, opt.warp_height,
                                                     opt.target_tag_px, opt.warp_center_x_ratio, opt.warp_center_y_ratio,
                                                     locked_pack, err)) {
                    locked = true;
                    locked_det = cur;
                }
                err.clear();
            }
        }

        if (opt.roi_mode == "fixed") {
            if (key == '1') selected = 0;
            if (key == '2') selected = 1;
            if (key == ',') size_step = std::max(0.001, size_step * 0.5);
            if (key == '.') size_step = std::min(0.25, size_step * 2.0);
            if (key == '[') move_step = std::max(0.001, move_step * 0.5);
            if (key == ']') move_step = std::min(0.25, move_step * 2.0);
            if (locked) {
                if (selected == 0) adjust_roi(rois.red_roi, key, move_step, size_step);
                else adjust_roi(rois.image_roi, key, move_step, size_step);
            }
        } else {
            tune_dynamic_cfg(dyn, key, pixel_step);
            clamp_dynamic_cfg(dyn, {opt.warp_width, opt.warp_height});
        }

        if (key == 'o') {
            if (save_rois_yaml(opt.save_rois, rois)) std::cout << "\n[save] rois -> " << opt.save_rois << "\n";
        }
        if (key == 'y' && locked) {
            if (save_warp_package(opt.save_warp, locked_pack)) std::cout << "\n[save] warp -> " << opt.save_warp << "\n";
        }
        if (key == 'p' && locked) {
            save_rois_yaml(opt.save_rois, rois);
            save_warp_package(opt.save_warp, locked_pack);
            write_report_md(opt.save_report, "Calibration Report", nullptr, build_effective_config(opt));
        }
    }

    release_model_runtime();
    if (opt.ui) cv::destroyAllWindows();
    return err.empty();
}

inline bool run_deploy(const AppOptions& opt_in, std::string& err) {
    AppOptions opt = opt_in;
    WarpPackage pack;
    if (!load_warp_package(opt.load_warp, pack)) {
        err = "failed to load warp package: " + opt.load_warp;
        return false;
    }
    RoiConfig rois = opt.default_rois;
    (void)load_rois_yaml(opt.load_rois, rois);

    int cam_w = opt.width, cam_h = opt.height;
    cv::VideoCapture cap;
    if (!open_capture(cap, opt.device, cam_w, cam_h, opt.fps, opt.fourcc, opt.buffer_size, opt.camera_soft_max, err)) return false;
    DynamicRedStackedConfig dyn = opt.dyn_cfg;
    clamp_dynamic_cfg(dyn, pack.warp_size);

    TextConsole console;
    console.enabled = opt.text_console && opt.ui;
    if (opt.ui) {
        cv::namedWindow("vision_app_camera", cv::WINDOW_NORMAL);
        cv::namedWindow("vision_app_warp", cv::WINDOW_NORMAL);
        if (opt.red_show_mask_window) cv::namedWindow("vision_app_red_mask", cv::WINDOW_NORMAL);
        console.open();
    }

    const std::string cfg_warn = build_deploy_config_warning(opt, pack, cam_w, cam_h);
    if (!cfg_warn.empty()) std::cerr << cfg_warn;

    cv::Mat frame, warped, valid, camera_show, warp_show;
    RoiRuntimeData roi_info;
    int stable_counter = 0;
    int frame_idx = 0;
    auto loop_t0 = std::chrono::steady_clock::now();

    while (true) {
        if (!grab_latest_frame(cap, opt.latest_only, opt.drain_grabs, frame)) { err = "failed to read frame"; break; }
        if (!apply_warp(frame, pack, warped, &valid)) { err = "failed to apply warp"; break; }
        camera_show = frame.clone();
        warp_show = warped.clone();

        std::string rerr;
        if (opt.roi_mode == "fixed") {
            if (extract_runtime_rois_fixed(warped, valid, rois, opt.red_cfg, roi_info, rerr)) draw_rois(warp_show, rois, -1);
        } else {
            if (extract_runtime_rois_dynamic_stacked(warped, valid, dyn, opt.red_cfg, stable_counter, roi_info, rerr)) {
                stable_counter = roi_info.trigger_ready ? std::min(stable_counter + 1, dyn.stable_frames_required) : 0;
                draw_dynamic_overlay(warp_show, roi_info);
                if (opt.red_show_mask_window && !roi_info.red_mask.empty()) cv::imshow("vision_app_red_mask", roi_info.red_mask);
            }
        }
        if (!rerr.empty()) cv::putText(warp_show, rerr, {12, warp_show.rows - 12}, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0,0,255), 1);
        if (opt.save_every_n > 0 && frame_idx % opt.save_every_n == 0) {
            if (opt.run_image_roi) save_crop_if_needed(opt.save_image_roi_dir, "image_roi", frame_idx, roi_info.image_bgr);
            if (opt.run_red) save_crop_if_needed(opt.save_red_roi_dir, "red_roi", frame_idx, roi_info.red_bgr);
        }

        if (opt.ui) {
            cv::imshow("vision_app_camera", downscale_for_preview(camera_show, opt.camera_preview_max));
            cv::imshow("vision_app_warp", downscale_for_preview(warp_show, opt.warp_preview_max));
            if (console.enabled) {
                if (opt.roi_mode == "dynamic_red_stacked") console.show(build_dynamic_console_lines(opt, roi_info, stable_counter));
                else console.show({"mode=deploy roi_mode=fixed", cv::format("red_ratio=%.4f", roi_info.red_ratio)});
            }
        }

        ++frame_idx;
        int key = opt.ui ? (cv::waitKey(1) & 0xFF) : -1;
        if (key == 27 || key == 'q') break;
        if (!opt.ui && opt.duration > 0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - loop_t0).count() >= opt.duration) break;
    }

    release_model_runtime();
    if (opt.ui) cv::destroyAllWindows();
    return err.empty();
}

} // namespace vision_app
