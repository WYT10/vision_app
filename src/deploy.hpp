#pragma once

#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "camera.hpp"
#include "calibrate.hpp"
#include "stats.hpp"
#include "model.hpp"

namespace vision_app {

struct RedThresholdConfig {
    int h1_low = 0;
    int h1_high = 10;
    int h2_low = 170;
    int h2_high = 180;
    int s_min = 80;
    int v_min = 60;
};

struct ModelConfig {
    bool enable = false;
    std::string backend = "off";   // off | onnx | ncnn
    std::string onnx_path;
    std::string ncnn_param_path;
    std::string ncnn_bin_path;
    std::string labels_path;
    int input_width = 128;
    int input_height = 128;
    std::string preprocess = "crop"; // crop | stretch | letterbox
    int threads = 4;
    int stride = 5;
    int topk = 5;
};

struct ModelResult {
    bool ran = false;
    bool ok = false;
    int top_index = -1;
    float top_score = 0.0f;
    std::string top_label;
    std::vector<ModelHit> topk;
    double infer_ms = 0.0;
    std::string summary;
};

struct RoiRuntimeData {
    cv::Mat red_bgr;
    cv::Mat red_mask;
    cv::Mat image_bgr;
    cv::Mat image_mask;
    double red_ratio = 0.0;
    int red_valid_pixels = 0;
    int image_valid_pixels = 0;
};

bool extract_runtime_rois(const cv::Mat& warped,
                          const cv::Mat& valid_mask,
                          const RoiConfig& rois,
                          const RedThresholdConfig& red_cfg,
                          RoiRuntimeData& out,
                          std::string& err);

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

    int camera_soft_max = 1600;
    int camera_preview_max = 640;
    int warp_preview_max = 640;
    int temp_preview_stride = 2;

    int warp_width = 384;
    int warp_height = 384;
    int target_tag_px = 128;

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
    ModelConfig model_cfg;
};

inline void print_calibrate_controls() {
    std::cout << R"TXT(
=== vision_app calibrate controls ===
Keys
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
  h             : show this help again
  q / ESC       : quit

Windows
  vision_app_camera : raw feed + tag overlay
  vision_app_warp   : warp preview + ROIs + invalid area white-filled
)TXT" << std::flush;
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
            << ". Deploy warp may be invalid.\n";
        any = true;
    }
    if ((opt.warp_width > 0 && opt.warp_width != pack.warp_size.width) ||
        (opt.warp_height > 0 && opt.warp_height != pack.warp_size.height) ||
        (opt.target_tag_px > 0 && opt.target_tag_px != pack.target_tag_px)) {
        oss << "[warn] deploy uses saved warp package values, not current CLI calibration hints"
            << " (cli warp=" << opt.warp_width << 'x' << opt.warp_height
            << " tag_px=" << opt.target_tag_px << ").\n";
        any = true;
    }
    return any ? oss.str() : std::string();
}

inline bool build_preview_from_detection(const cv::Mat& frame,
                                         const AprilTagDetection& det,
                                         const AppOptions& opt,
                                         cv::Mat& preview,
                                         std::string& err) {
    preview.release();
    if (!det.found) { err.clear(); return false; }
    WarpPackage tmp;
    if (!build_centered_warp_package_from_detection_px(det, frame.size(), opt.warp_width, opt.warp_height,
                                                       opt.target_tag_px, tmp, err)) {
        return false;
    }
    cv::Mat valid;
    if (!apply_warp(frame, tmp, preview, &valid)) {
        err = "failed to build preview warp";
        return false;
    }
    return true;
}

inline bool run_calibrate(const AppOptions& opt, std::string& err) {
    int cam_w = opt.width, cam_h = opt.height;
    cv::VideoCapture cap;
    if (!open_capture(cap, opt.device, cam_w, cam_h, opt.fps, opt.fourcc, opt.buffer_size, opt.camera_soft_max, err)) return false;

    RoiConfig rois = opt.default_rois;
    (void)load_rois_yaml(opt.load_rois, rois);

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
    int frame_idx = 0;
    AprilTagDetection cur, locked_det;
    WarpPackage locked_pack;
    cv::Mat frame, camera_show, warp_show, temp_preview, warped, valid;

    cv::namedWindow("vision_app_camera", cv::WINDOW_NORMAL);
    cv::namedWindow("vision_app_warp", cv::WINDOW_NORMAL);
    cv::resizeWindow("vision_app_camera", opt.camera_preview_max, std::max(240, opt.camera_preview_max * 3 / 4));
    cv::resizeWindow("vision_app_warp", opt.warp_preview_max, opt.warp_preview_max);
    print_calibrate_controls();

    while (true) {
        if (!grab_latest_frame(cap, opt.latest_only, opt.drain_grabs, frame)) { err = "failed to read frame"; break; }

        if (!locked) {
            std::string derr;
            detect_apriltag_best(frame, tag_cfg, cur, derr);
            const bool stable = locker.update(cur);
            if (!opt.manual_lock_only && stable && cur.found) {
                if (build_centered_warp_package_from_detection_px(cur, frame.size(), opt.warp_width, opt.warp_height,
                                                                 opt.target_tag_px, locked_pack, err)) {
                    locked = true;
                    locked_det = cur;
                    std::cout << "\n[lock] auto family=" << locked_det.family << " id=" << locked_det.id << "\n";
                }
                err.clear();
            }

            camera_show = frame.clone();
            draw_detection_overlay(camera_show, cur);
            cv::putText(camera_show, cur.found ? "SEARCH" : "SEARCH: no tag", {12,56},
                        cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(0,255,255), 2);
            camera_show = downscale_for_preview(camera_show, opt.camera_preview_max);

            if (cur.found && (frame_idx % std::max(1, opt.temp_preview_stride) == 0)) {
                std::string werr;
                if (build_preview_from_detection(frame, cur, opt, temp_preview, werr)) {
                    draw_rois(temp_preview, rois, selected);
                    cv::putText(temp_preview,
                                "preview warp=" + std::to_string(opt.warp_width) + "x" + std::to_string(opt.warp_height) +
                                " tag_px=" + std::to_string(opt.target_tag_px),
                                {12,24}, cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(0,100,0), 2);
                }
            }
            warp_show = temp_preview.empty() ? cv::Mat(std::max(64,opt.warp_height), std::max(64,opt.warp_width), CV_8UC3, cv::Scalar(235,235,235))
                                             : temp_preview.clone();
            if (temp_preview.empty()) {
                cv::putText(warp_show, "waiting for tag", {16, std::max(32, warp_show.rows/2)}, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(60,60,60), 2);
            }
            warp_show = downscale_for_preview(warp_show, opt.warp_preview_max);
        } else {
            cur = locked_det;
            if (!apply_warp(frame, locked_pack, warped, &valid)) { err = "failed to apply locked warp"; break; }
            camera_show = frame.clone();
            draw_detection_overlay(camera_show, locked_det);
            cv::putText(camera_show, "LOCKED", {12,56}, cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(0,255,0), 2);
            camera_show = downscale_for_preview(camera_show, opt.camera_preview_max);

            warp_show = warped.clone();
            draw_rois(warp_show, rois, selected);
            cv::putText(warp_show,
                        "LOCKED warp=" + std::to_string(locked_pack.warp_size.width) + "x" + std::to_string(locked_pack.warp_size.height) +
                        " tag_px=" + std::to_string(locked_pack.target_tag_px),
                        {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.60, cv::Scalar(0,120,0), 2);
            warp_show = downscale_for_preview(warp_show, opt.warp_preview_max);
        }

        cv::imshow("vision_app_camera", camera_show);
        cv::imshow("vision_app_warp", warp_show);

        ++frame_idx;
        const int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') break;
        if (key == 'u') {
            locked = false;
            locker.reset();
            locked_pack = {};
            temp_preview.release();
            std::cout << "\n[lock] released\n";
        }
        if (key == ' ' || key == 13) {
            if (cur.found) {
                if (build_centered_warp_package_from_detection_px(cur, frame.size(), opt.warp_width, opt.warp_height,
                                                                 opt.target_tag_px, locked_pack, err)) {
                    locked = true;
                    locked_det = cur;
                    std::cout << "\n[lock] manual family=" << locked_det.family << " id=" << locked_det.id << "\n";
                } else {
                    std::cerr << "\n[lock] rejected: " << err << "\n";
                    err.clear();
                }
            }
        }
        if (key == '1') { selected = 0; std::cout << "\n[roi] selected red_roi\n"; }
        if (key == '2') { selected = 1; std::cout << "\n[roi] selected image_roi\n"; }
        if (key == '[') { move_step = std::max(0.001, move_step * 0.5); std::cout << "\n[step] move_step=" << move_step << "\n"; }
        if (key == ']') { move_step = std::min(0.25, move_step * 2.0); std::cout << "\n[step] move_step=" << move_step << "\n"; }
        if (key == ',') { size_step = std::max(0.001, size_step * 0.5); std::cout << "\n[step] size_step=" << size_step << "\n"; }
        if (key == '.') { size_step = std::min(0.25, size_step * 2.0); std::cout << "\n[step] size_step=" << size_step << "\n"; }
        if (key == 'h') { print_calibrate_controls(); }
        if (key == 'r') { rois = opt.default_rois; std::cout << "\n[roi] reset to defaults\n"; }
        if (locked) {
            if (selected == 0) adjust_roi(rois.red_roi, key, move_step, size_step);
            else adjust_roi(rois.image_roi, key, move_step, size_step);
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
            write_report_md(opt.save_report, "Calibration Report", nullptr,
                            std::string("- Locked family: ") + locked_pack.family + "\n" +
                            "- Locked id: " + std::to_string(locked_pack.id) + "\n" +
                            "- Warp size: " + std::to_string(locked_pack.warp_size.width) + "x" + std::to_string(locked_pack.warp_size.height) + "\n" +
                            "- Target tag px: " + std::to_string(locked_pack.target_tag_px) + "\n" +
                            "- red_roi ratio: " + std::to_string(rois.red_roi.x) + "," + std::to_string(rois.red_roi.y) + "," + std::to_string(rois.red_roi.w) + "," + std::to_string(rois.red_roi.h) + "\n" +
                            "- image_roi ratio: " + std::to_string(rois.image_roi.x) + "," + std::to_string(rois.image_roi.y) + "," + std::to_string(rois.image_roi.w) + "," + std::to_string(rois.image_roi.h));
            std::cout << "\n[save] all -> " << opt.save_warp << " ; " << opt.save_rois << " ; " << opt.save_report << "\n";
        }
    }

    cv::destroyAllWindows();
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
    (void)load_rois_yaml(opt.load_rois, rois);

    bool model_ready = false;
    if (opt.model_cfg.enable && opt.model_cfg.backend != "off") {
        std::string merr;
        model_ready = init_model_runtime(opt.model_cfg, merr);
        if (!model_ready) std::cerr << "\n[model] disabled: " << merr << "\n";
    }

    int cam_w = opt.width, cam_h = opt.height;
    cv::VideoCapture cap;
    if (!open_capture(cap, opt.device, cam_w, cam_h, opt.fps, opt.fourcc, opt.buffer_size, opt.camera_soft_max, err)) return false;

    const std::string start_warn = build_deploy_config_warning(opt, pack, cam_w, cam_h);
    if (!start_warn.empty()) std::cerr << start_warn;

    cv::Mat frame, warped, valid, camera_show, warp_show;
    RoiRuntimeData roi_info;
    ModelResult model_res;
    cv::namedWindow("vision_app_camera", cv::WINDOW_NORMAL);
    cv::namedWindow("vision_app_warp", cv::WINDOW_NORMAL);
    cv::resizeWindow("vision_app_camera", opt.camera_preview_max, std::max(240, opt.camera_preview_max * 3 / 4));
    cv::resizeWindow("vision_app_warp", opt.warp_preview_max, opt.warp_preview_max);

    std::cout << "\n=== vision_app deploy ===\n"
              << "loaded warp: " << opt.load_warp << "\n"
              << "loaded rois: " << opt.load_rois << "\n"
              << "backend: " << (model_ready ? model_runtime_backend() : std::string("off")) << "\n";

    int frame_idx = 0;
    using clk = std::chrono::steady_clock;
    while (true) {
        auto t0 = clk::now();
        if (!grab_latest_frame(cap, opt.latest_only, opt.drain_grabs, frame)) { err = "failed to read frame"; break; }
        auto t1 = clk::now();
        if (!apply_warp(frame, pack, warped, &valid)) {
            err = "failed to apply warp; check live camera mode vs saved calibration source size";
            break;
        }
        auto t2 = clk::now();
        camera_show = downscale_for_preview(frame, opt.camera_preview_max);
        warp_show = warped.clone();
        draw_rois(warp_show, rois, -1);
        cv::putText(warp_show, "DEPLOY family=" + pack.family + " id=" + std::to_string(pack.id), {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.68, cv::Scalar(0,120,0), 2);

        std::string rerr;
        if (extract_runtime_rois(warped, valid, rois, opt.red_cfg, roi_info, rerr)) {
            auto t3 = clk::now();
            cv::putText(warp_show, "red_ratio=" + std::to_string(roi_info.red_ratio).substr(0,5), {12, 56}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0,120,0), 2);
            if (model_ready && (frame_idx % std::max(1, opt.model_cfg.stride) == 0)) {
                std::string merr;
                run_model_on_image_roi(roi_info, opt.model_cfg, model_res, merr);
            }
            auto t4 = clk::now();
            if (model_res.ran) {
                cv::putText(warp_show, model_res.summary, {12, 80}, cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(0,120,0), 1);
            }
            const double cap_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            const double warp_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            const double roi_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
            const double model_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
            cv::putText(warp_show,
                        "cap=" + std::to_string(cap_ms).substr(0,5) + " warp=" + std::to_string(warp_ms).substr(0,5) +
                        " roi=" + std::to_string(roi_ms).substr(0,5) + " model=" + std::to_string(model_ms).substr(0,5),
                        {12, 104}, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0,120,0), 1);
        }
        if (!start_warn.empty()) {
            cv::putText(warp_show, "WARN: live camera args differ from saved calibration", {12, std::max(124, warped.rows - 18)},
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0,0,255), 1);
        }
        warp_show = downscale_for_preview(warp_show, opt.warp_preview_max);
        cv::imshow("vision_app_camera", camera_show);
        cv::imshow("vision_app_warp", warp_show);
        ++frame_idx;
        const int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') break;
    }

    release_model_runtime();
    cv::destroyAllWindows();
    std::cout << "\n";
    return err.empty();
}

} // namespace vision_app
