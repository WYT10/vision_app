
#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include "camera.hpp"
#include "calibrate.hpp"
#include "stats.hpp"
#include "model.hpp"

namespace vision_app {

struct AppOptions {
    std::string mode = "probe"; // probe | calibrate | deploy
    std::string device = "/dev/video0";
    int width = 640;
    int height = 480;
    int fps = 120;
    std::string fourcc = "MJPG";
    int buffer_size = 1;
    bool latest_only = true;
    int drain_grabs = 1;
    bool headless = false;
    int duration = 5;

    bool ui = true;
    bool debug = true;
    bool save_roi_frames = false;
    std::string save_roi_dir = "../report/roi_snaps";

    int camera_soft_max = 1000;
    int camera_preview_max = 640;
    int warp_preview_max = 640;

    AprilTagConfig tag_cfg;
    int warp_width = 384;
    int warp_height = 384;
    int target_tag_px = 128;

    std::string save_warp = "../report/warp_package.yml.gz";
    std::string load_warp = "../report/warp_package.yml.gz";
    std::string save_rois = "../report/rois.yml";
    std::string load_rois = "../report/rois.yml";
    std::string save_report = "../report/latest_report.md";

    RoiConfig default_rois;
    RedThresholdConfig red_cfg;
    ModelConfig model_cfg;
};

bool extract_runtime_rois(const cv::Mat& warped,
                          const cv::Mat& valid_mask,
                          const RoiConfig& rois,
                          const RedThresholdConfig& red_cfg,
                          RoiRuntimeData& out,
                          std::string& err);

inline void print_calibrate_controls() {
    std::cout << R"TXT(
=== vision_app calibrate controls ===
SPACE / ENTER : lock current tag
u             : unlock / reacquire
1 / 2         : select red_roi / image_roi
w a s d       : move ROI
i / k         : height - / +
j / l         : width  - / +
[ / ]         : move step down / up
, / .         : size step down / up
p             : save all
y             : save warp only
o             : save rois only
r             : reset rois
h             : show help
q / ESC       : quit
)TXT" << std::flush;
}

inline cv::Mat make_blank_preview(int w, int h, const std::string& text) {
    cv::Mat img(std::max(64, h), std::max(64, w), CV_8UC3, cv::Scalar(235,235,235));
    cv::putText(img, text, {16, img.rows / 2}, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(60,60,60), 2);
    return img;
}

inline void maybe_save_roi_snapshot(const AppOptions& opt,
                                    int frame_idx,
                                    const RoiRuntimeData& roi_info,
                                    const ModelResult& model_res) {
    if (!opt.save_roi_frames || roi_info.image_bgr.empty()) return;
    std::filesystem::create_directories(opt.save_roi_dir);
    const std::string label = model_res.best.label.empty() ? "unknown" : model_res.best.label;
    const auto path = std::filesystem::path(opt.save_roi_dir) /
                      (cv::format("f%06d_%s.jpg", frame_idx, label.c_str()));
    cv::imwrite(path.string(), roi_info.image_bgr);
}

inline bool run_probe(const AppOptions& opt, std::string& err) {
    CameraProbeResult probe;
    if (!probe_camera(opt.device, probe, err)) return false;
    print_probe(probe);
    RuntimeStats stats;
    if (!bench_capture(opt.device, opt.width, opt.height, opt.fps, opt.fourcc,
                       opt.buffer_size, opt.latest_only, opt.drain_grabs,
                       opt.headless, opt.duration, opt.camera_soft_max,
                       opt.camera_preview_max, stats, err)) return false;
    print_runtime_stats(stats);
    std::ostringstream extra;
    extra << "## Requested mode\n\n"
          << "- Width: " << opt.width << "\n"
          << "- Height: " << opt.height << "\n"
          << "- FPS: " << opt.fps << "\n"
          << "- FOURCC: " << opt.fourcc << "\n";
    (void)write_report_md(opt.save_report, "vision_app probe report", &stats, extra.str());
    return true;
}

inline bool run_calibrate(const AppOptions& opt, std::string& err) {
    int cam_w = opt.width, cam_h = opt.height;
    cv::VideoCapture cap;
    if (!open_capture(cap, opt.device, cam_w, cam_h, opt.fps, opt.fourcc, opt.buffer_size, opt.camera_soft_max, err)) return false;

    RoiConfig rois = opt.default_rois;
    (void)load_rois_yaml(opt.load_rois, rois);

    TagLocker locker(opt.tag_cfg.lock_frames);
    bool locked = false;
    int selected = 0;
    double move_step = 0.01;
    double size_step = 0.01;
    int frame_idx = 0;
    AprilTagDetection cur, locked_det;
    WarpPackage locked_pack;
    RoiRuntimeData roi_info;
    ModelResult model_res;
    cv::Mat frame, camera_show, warp_show, temp_preview, warped, valid;

    if (opt.ui && !opt.headless) {
        cv::namedWindow("vision_app_camera", cv::WINDOW_NORMAL);
        cv::namedWindow("vision_app_warp", cv::WINDOW_NORMAL);
        cv::resizeWindow("vision_app_camera", opt.camera_preview_max, std::max(240, opt.camera_preview_max * 3 / 4));
        cv::resizeWindow("vision_app_warp", opt.warp_preview_max, opt.warp_preview_max);
    }
    if (opt.debug) print_calibrate_controls();

    while (true) {
        if (!grab_latest_frame(cap, opt.latest_only, opt.drain_grabs, frame)) { err = "failed to read frame"; break; }

        if (!locked) {
            std::string derr;
            detect_apriltag_best(frame, opt.tag_cfg, cur, derr);
            (void)locker.update(cur);

            camera_show = frame.clone();
            draw_detection_overlay(camera_show, cur);
            cv::putText(camera_show, cur.found ? "CALIBRATE: press SPACE to lock" : "CALIBRATE: no tag",
                        {12, 56}, cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(0,255,255), 2);
            camera_show = downscale_for_preview(camera_show, opt.camera_preview_max);

            if (cur.found) {
                WarpPackage temp_pack;
                std::string werr;
                if (build_centered_warp_package_from_detection_px(cur, frame.size(), opt.warp_width, opt.warp_height,
                                                                 opt.target_tag_px, temp_pack, werr)) {
                    if (apply_warp(frame, temp_pack, temp_preview, nullptr)) {
                        draw_rois(temp_preview, rois, -1);
                        cv::putText(temp_preview,
                                    cv::format("preview warp=%dx%d tag_px=%d", temp_pack.warp_size.width, temp_pack.warp_size.height, temp_pack.target_tag_px),
                                    {12, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(0, 100, 0), 2);
                    }
                }
            }
            if (temp_preview.empty()) warp_show = make_blank_preview(opt.warp_width, opt.warp_height, "waiting for tag");
            else warp_show = temp_preview.clone();
            warp_show = downscale_for_preview(warp_show, opt.warp_preview_max);
        } else {
            if (!apply_warp(frame, locked_pack, warped, &valid)) { err = "failed to apply locked warp"; break; }
            camera_show = frame.clone();
            draw_detection_overlay(camera_show, locked_det);
            cv::putText(camera_show, "LOCKED", {12, 56}, cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(0,255,0), 2);
            camera_show = downscale_for_preview(camera_show, opt.camera_preview_max);

            warp_show = warped.clone();
            draw_rois(warp_show, rois, selected);
            cv::putText(warp_show,
                        cv::format("LOCKED warp=%dx%d tag_px=%d", locked_pack.warp_size.width, locked_pack.warp_size.height, locked_pack.target_tag_px),
                        {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.60, cv::Scalar(0,120,0), 2);
            const cv::Rect ir = roi_to_rect(rois.image_roi, warped.size());
            cv::putText(warp_show,
                        cv::format("image_roi=%dx%d", ir.width, ir.height),
                        {12, 56}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0,120,0), 2);
            warp_show = downscale_for_preview(warp_show, opt.warp_preview_max);
        }

        if (opt.ui && !opt.headless) {
            cv::imshow("vision_app_camera", camera_show.empty() ? frame : camera_show);
            cv::imshow("vision_app_warp", warp_show);
        }
        ++frame_idx;
        const int key = opt.headless ? -1 : (cv::waitKey(1) & 0xFF);
        if (key == 27 || key == 'q') break;
        if (key == 'h') print_calibrate_controls();
        if (!locked && (key == ' ' || key == 13) && cur.found) {
            if (build_centered_warp_package_from_detection_px(cur, frame.size(), opt.warp_width, opt.warp_height,
                                                             opt.target_tag_px, locked_pack, err)) {
                locked = true;
                locked_det = cur;
                if (opt.debug) std::cout << "[lock] family=" << locked_det.family << " id=" << locked_det.id
                                         << " warp=" << locked_pack.warp_size.width << 'x' << locked_pack.warp_size.height
                                         << " tag_px=" << locked_pack.target_tag_px << "\n";
            }
        }
        if (locked && key == 'u') locked = false;
        if (key == '1') selected = 0;
        if (key == '2') selected = 1;
        RoiRatio* rr = (selected == 0) ? &rois.red_roi : &rois.image_roi;
        if (key == '[') move_step = std::max(0.002, move_step * 0.5);
        if (key == ']') move_step = std::min(0.10, move_step * 2.0);
        if (key == ',') size_step = std::max(0.002, size_step * 0.5);
        if (key == '.') size_step = std::min(0.10, size_step * 2.0);
        if (key == 'w') rr->y -= move_step;
        if (key == 's') rr->y += move_step;
        if (key == 'a') rr->x -= move_step;
        if (key == 'd') rr->x += move_step;
        if (key == 'j') rr->w -= size_step;
        if (key == 'l') rr->w += size_step;
        if (key == 'i') rr->h -= size_step;
        if (key == 'k') rr->h += size_step;
        *rr = clamp_roi(*rr);
        if (key == 'r') rois = opt.default_rois;
        if (key == 'y' || key == 'p') (void)save_warp_package(opt.save_warp, locked_pack);
        if (key == 'o' || key == 'p') (void)save_rois_yaml(opt.save_rois, rois);
    }
    if (!opt.headless) cv::destroyAllWindows();
    return err.empty();
}

inline bool run_deploy(const AppOptions& opt, std::string& err) {
    WarpPackage pack;
    if (!load_warp_package(opt.load_warp, pack)) { err = "failed to load warp package: " + opt.load_warp; return false; }
    RoiConfig rois = opt.default_rois;
    (void)load_rois_yaml(opt.load_rois, rois);

    bool model_ready = false;
    if (opt.model_cfg.enable && opt.model_cfg.backend != "off") {
        std::string merr;
        model_ready = init_model_runtime(opt.model_cfg, merr);
        if (!model_ready && opt.debug) std::cerr << "[model] disabled: " << merr << "\n";
    }

    int cam_w = opt.width, cam_h = opt.height;
    cv::VideoCapture cap;
    if (!open_capture(cap, opt.device, cam_w, cam_h, opt.fps, opt.fourcc, opt.buffer_size, opt.camera_soft_max, err)) return false;

    cv::Mat frame, warped, valid, camera_show, warp_show;
    RoiRuntimeData roi_info;
    ModelResult model_res;

    if (opt.ui && !opt.headless) {
        cv::namedWindow("vision_app_camera", cv::WINDOW_NORMAL);
        cv::namedWindow("vision_app_warp", cv::WINDOW_NORMAL);
        cv::resizeWindow("vision_app_camera", opt.camera_preview_max, std::max(240, opt.camera_preview_max * 3 / 4));
        cv::resizeWindow("vision_app_warp", opt.warp_preview_max, opt.warp_preview_max);
    }

    using clk = std::chrono::steady_clock;
    auto t_start = clk::now();
    int frame_idx = 0;
    RuntimeStats stats{};
    while (true) {
        const auto f0 = clk::now();
        if (!grab_latest_frame(cap, opt.latest_only, opt.drain_grabs, frame)) { err = "failed to read frame"; break; }
        const auto t0 = clk::now();
        if (!apply_warp(frame, pack, warped, &valid)) { err = "failed to apply warp"; break; }
        const auto t1 = clk::now();
        std::string rerr;
        if (!extract_runtime_rois(warped, valid, rois, opt.red_cfg, roi_info, rerr)) { err = rerr; break; }
        const auto t2 = clk::now();
        if (model_ready && (frame_idx % std::max(1, opt.model_cfg.stride) == 0)) {
            std::string merr;
            (void)run_model_on_image_roi(roi_info, opt.model_cfg, model_res, merr);
            maybe_save_roi_snapshot(opt, frame_idx, roi_info, model_res);
        }
        const auto t3 = clk::now();

        if (opt.ui && !opt.headless) {
            camera_show = downscale_for_preview(frame, opt.camera_preview_max);
            warp_show = warped.clone();
            draw_rois(warp_show, rois, -1);
            cv::putText(warp_show, cv::format("red_ratio=%.3f", roi_info.red_ratio), {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.60, cv::Scalar(0,120,0), 2);
            cv::putText(warp_show, cv::format("warp %.2fms roi %.2fms infer %.2fms", 
                std::chrono::duration<double,std::milli>(t1-t0).count(),
                std::chrono::duration<double,std::milli>(t2-t1).count(),
                model_res.infer_ms), {12, 56}, cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(0,120,0), 1);
            if (model_res.ran) cv::putText(warp_show, model_res.summary, {12, 80}, cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(0,120,0), 1);
            warp_show = downscale_for_preview(warp_show, opt.warp_preview_max);
            cv::imshow("vision_app_camera", camera_show);
            cv::imshow("vision_app_warp", warp_show);
        }

        ++frame_idx;
        ++stats.frames;
        const double total_ms = std::chrono::duration<double,std::milli>(t3 - f0).count();
        if (frame_idx > 1) {
            stats.frame_time_avg_ms += total_ms;
            if (stats.frame_time_min_ms == 0.0) stats.frame_time_min_ms = total_ms;
            stats.frame_time_min_ms = std::min(stats.frame_time_min_ms, total_ms);
            stats.frame_time_max_ms = std::max(stats.frame_time_max_ms, total_ms);
        }
        const int key = opt.headless ? -1 : (cv::waitKey(1) & 0xFF);
        if (key == 27 || key == 'q') break;
    }
    stats.elapsed_sec = std::chrono::duration<double>(clk::now() - t_start).count();
    if (stats.elapsed_sec > 0.0) stats.fps_avg = static_cast<double>(stats.frames) / stats.elapsed_sec;
    stats.actual_width = frame.cols;
    stats.actual_height = frame.rows;
    if (stats.frames > 1) stats.frame_time_avg_ms /= static_cast<double>(stats.frames - 1);
    if (stats.frame_time_max_ms > 0.0) stats.fps_min = 1000.0 / stats.frame_time_max_ms;
    if (stats.frame_time_min_ms > 0.0) stats.fps_max = 1000.0 / stats.frame_time_min_ms;
    if (opt.debug) print_runtime_stats(stats);
    std::ostringstream extra;
    extra << "## Deploy\n\n"
          << "- Backend: " << opt.model_cfg.backend << "\n"
          << "- Warp: " << pack.warp_size.width << 'x' << pack.warp_size.height << "\n"
          << "- Target tag px: " << pack.target_tag_px << "\n"
          << "- Model input: " << opt.model_cfg.input_width << 'x' << opt.model_cfg.input_height << "\n"
          << "- Last model summary: " << model_res.summary << "\n";
    (void)write_report_md(opt.save_report, "vision_app deploy report", &stats, extra.str());
    if (!opt.headless) cv::destroyAllWindows();
    release_model_runtime();
    return err.empty();
}

} // namespace vision_app
