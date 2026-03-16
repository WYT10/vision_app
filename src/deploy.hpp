#pragma once

#include <algorithm>
#include <iostream>
#include <string>

#include "camera.hpp"
#include "calibrate.hpp"
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

struct ModelConfig {
    bool enable = false;
    std::string backend = "off";   // off | onnx | ncnn
    std::string path;
    int input_width = 224;
    int input_height = 224;
    bool swap_rb = true;
    double scale = 1.0 / 255.0;
    int stride = 5;
};

struct ModelResult {
    bool ran = false;
    bool ok = false;
    int top_index = -1;
    float top_score = 0.0f;
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

bool init_model_runtime(const ModelConfig& cfg, std::string& err);
void release_model_runtime();
bool extract_runtime_rois(const cv::Mat& warped,
                          const cv::Mat& valid_mask,
                          const RoiConfig& rois,
                          const RedThresholdConfig& red_cfg,
                          RoiRuntimeData& out,
                          std::string& err);
bool run_model_on_image_roi(const RoiRuntimeData& in,
                            const ModelConfig& cfg,
                            ModelResult& out,
                            std::string& err);

struct AppOptions {
    std::string mode = "probe";
    std::string device = "/dev/video0";
    int width = 640;
    int height = 480;
    int fps = 180;
    std::string fourcc = "MJPG";
    int buffer_size = 1;
    bool latest_only = true;
    int drain_grabs = 1;
    bool headless = false;
    int duration = 10;

    int camera_soft_max = 1000;
    int warp_soft_max = 700;
    int preview_soft_max = 500; // backward-compatible fallback
    int camera_preview_max = 640;
    int warp_preview_max = 640;
    int temp_preview_square = 260;
    int temp_preview_stride = 3;
    double tag_fill_ratio = 0.70;

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

inline void print_live_controls() {
    std::cout
        << "\n=== vision_app live controls ===\n"
        << "Build every time with:\n"
        << "  mkdir -p build\n"
        << "  cd build\n"
        << "  cmake ..\n"
        << "  make -j$(nproc)\n\n"
        << "Windows\n"
        << "  vision_app_camera : raw feed + tag overlay\n"
        << "  vision_app_warp   : centered warp + ROIs + empty white area\n\n"
        << "Search / lock\n"
        << "  SPACE / ENTER : lock current tag\n"
        << "  u             : unlock / reacquire\n"
        << "  q / ESC       : quit\n\n"
        << "ROI selection\n"
        << "  1             : select red_roi\n"
        << "  2             : select image_roi\n\n"
        << "ROI move\n"
        << "  w a s d       : up / left / down / right\n\n"
        << "ROI size\n"
        << "  i / k         : height - / +\n"
        << "  j / l         : width  - / +\n\n"
        << "Step control\n"
        << "  [ / ]         : move step down / up\n"
        << "  , / .         : size step down / up\n\n"
        << "Save\n"
        << "  p             : save all\n"
        << "  y             : save warp only\n"
        << "  o             : save rois only\n"
        << "  r             : reset rois\n"
        << std::flush;
}

inline void print_status_to_terminal(bool locked,
                                     const AprilTagDetection& cur,
                                     int selected,
                                     double move_step,
                                     double size_step,
                                     double fill_ratio,
                                     const RoiRuntimeData* roi_info,
                                     const ModelResult* model_res) {
    std::cout << "\r[" << (locked ? "LOCKED" : "SEARCH") << "] ";
    if (cur.found) {
        std::cout << "tag family=" << cur.family << " id=" << cur.id << "  ";
    } else {
        std::cout << "tag none  ";
    }
    std::cout << "roi=" << (selected == 0 ? "red" : "image")
              << " move=" << move_step
              << " size=" << size_step
              << " fill=" << fill_ratio;
    if (roi_info) {
        std::cout << " red_ratio=" << roi_info->red_ratio
                  << " red_valid=" << roi_info->red_valid_pixels
                  << " image_valid=" << roi_info->image_valid_pixels;
    }
    if (model_res && model_res->ran) {
        std::cout << " model=" << model_res->summary;
    }
    std::cout << "        " << std::flush;
}

inline cv::Mat make_blank_preview(int side, const std::string& text) {
    cv::Mat img(std::max(64, side), std::max(64, side), CV_8UC3, cv::Scalar(235,235,235));
    cv::putText(img, text, {16, img.rows / 2}, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(60,60,60), 2);
    return img;
}

inline bool build_preview_from_detection(const cv::Mat& frame,
                                         const AprilTagDetection& det,
                                         const AppOptions& opt,
                                         cv::Mat& preview,
                                         std::string& err) {
    preview.release();
    if (!det.found) {
        err.clear();
        return false;
    }
    WarpPackage temp_pack;
    if (!build_warp_package_from_detection(det,
                                           frame.size(),
                                           std::max(128, std::min(opt.temp_preview_square, opt.warp_soft_max)),
                                           opt.tag_fill_ratio,
                                           temp_pack,
                                           err)) {
        return false;
    }
    cv::Mat valid;
    if (!apply_warp(frame, temp_pack, preview, &valid)) {
        err = "failed to build preview warp";
        return false;
    }
    return true;
}

inline bool run_live(const AppOptions& opt, std::string& err) {
    int cam_w = opt.width, cam_h = opt.height;
    cv::VideoCapture cap;
    if (!open_capture(cap, opt.device, cam_w, cam_h, opt.fps, opt.fourcc, opt.buffer_size, opt.camera_soft_max, err)) return false;

    RoiConfig rois = opt.default_rois;
    (void)load_rois_yaml(opt.load_rois, rois);

    bool model_ready = false;
    if (opt.model_cfg.enable && opt.model_cfg.backend != "off" && !opt.model_cfg.path.empty()) {
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
    int frame_idx = 0;
    int last_print_tag = -999999;
    std::string last_print_family;
    AprilTagDetection cur, locked_det;
    WarpPackage locked_pack;
    ModelResult model_res;
    RoiRuntimeData roi_info;
    cv::Mat frame, camera_show, warp_show, temp_preview, warped, valid;

    cv::namedWindow("vision_app_camera", cv::WINDOW_NORMAL);
    cv::namedWindow("vision_app_warp", cv::WINDOW_NORMAL);
    cv::resizeWindow("vision_app_camera", opt.camera_preview_max, std::max(240, opt.camera_preview_max * 3 / 4));
    cv::resizeWindow("vision_app_warp", opt.warp_preview_max, opt.warp_preview_max);

    print_live_controls();

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
                if (build_warp_package_from_detection(cur, frame.size(), opt.warp_soft_max, opt.tag_fill_ratio, locked_pack, err)) {
                    locked = true;
                    locked_det = cur;
                    std::cout << "\n[lock] auto lock family=" << locked_det.family << " id=" << locked_det.id << "\n";
                }
                err.clear();
            }

            camera_show = frame.clone();
            draw_detection_overlay(camera_show, cur);
            cv::putText(camera_show,
                        cur.found ? "SEARCH: centered live warp in other window" : "SEARCH: no tag",
                        {12, 56}, cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(0,255,255), 2);
            camera_show = downscale_for_preview(camera_show, opt.camera_preview_max);

            if (cur.found && (frame_idx % std::max(1, opt.temp_preview_stride) == 0)) {
                std::string werr;
                if (build_preview_from_detection(frame, cur, opt, temp_preview, werr)) {
                    const cv::Rect rr = roi_to_rect(rois.red_roi, temp_preview.size());
                    const cv::Rect ir = roi_to_rect(rois.image_roi, temp_preview.size());
                    cv::rectangle(temp_preview, rr, cv::Scalar(0,0,255), 2);
                    cv::rectangle(temp_preview, ir, cv::Scalar(255,0,0), 2);
                    cv::putText(temp_preview, "LIVE centered warp preview", {12, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(0, 100, 0), 2);
                }
            }
            if (temp_preview.empty()) {
                warp_show = make_blank_preview(opt.temp_preview_square, "waiting for tag");
            } else {
                warp_show = temp_preview.clone();
            }
            warp_show = downscale_for_preview(warp_show, opt.warp_preview_max);
        } else {
            cur = locked_det;
            if (!apply_warp(frame, locked_pack, warped, &valid)) {
                err = "failed to apply locked warp";
                break;
            }
            camera_show = frame.clone();
            draw_detection_overlay(camera_show, locked_det);
            cv::putText(camera_show, "LOCKED", {12, 56}, cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(0,255,0), 2);
            camera_show = downscale_for_preview(camera_show, opt.camera_preview_max);

            warp_show = warped.clone();
            draw_rois(warp_show, rois, selected);
            cv::putText(warp_show,
                        "LOCKED centered warp family=" + locked_pack.family + " id=" + std::to_string(locked_pack.id),
                        {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.60, cv::Scalar(0,120,0), 2);

            std::string rerr;
            if (extract_runtime_rois(warped, valid, rois, opt.red_cfg, roi_info, rerr)) {
                if (model_ready && (frame_idx % std::max(1, opt.model_cfg.stride) == 0)) {
                    std::string merr;
                    if (!run_model_on_image_roi(roi_info, opt.model_cfg, model_res, merr)) {
                        model_res = {};
                        model_res.ran = true;
                        model_res.summary = "model err";
                    }
                }
                cv::putText(warp_show,
                            "red_ratio=" + std::to_string(roi_info.red_ratio).substr(0,5),
                            {12, 56}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0,120,0), 2);
                if (model_res.ran) {
                    cv::putText(warp_show, model_res.summary, {12, 80}, cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(0,120,0), 1);
                }
            }
            warp_show = downscale_for_preview(warp_show, opt.warp_preview_max);
        }

        cv::imshow("vision_app_camera", camera_show);
        cv::imshow("vision_app_warp", warp_show);

        if (cur.found && (cur.id != last_print_tag || cur.family != last_print_family)) {
            std::cout << "\n[tag] family=" << cur.family << " id=" << cur.id << (locked ? " [locked]" : "") << "\n";
            last_print_tag = cur.id;
            last_print_family = cur.family;
        }
        print_status_to_terminal(locked, cur, selected, move_step, size_step, opt.tag_fill_ratio, locked ? &roi_info : nullptr, locked ? &model_res : nullptr);

        ++frame_idx;
        const int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') break;
        if (key == 'u') {
            locked = false;
            locker.reset();
            locked_pack = {};
            temp_preview.release();
            model_res = {};
            std::cout << "\n[lock] released\n";
        }
        if (key == ' ' || key == 13) {
            if (cur.found) {
                if (build_warp_package_from_detection(cur, frame.size(), opt.warp_soft_max, opt.tag_fill_ratio, locked_pack, err)) {
                    locked = true;
                    locked_det = cur;
                    std::cout << "\n[lock] manual lock family=" << locked_det.family << " id=" << locked_det.id << "\n";
                } else {
                    std::cerr << "\n[lock] rejected: " << err << "\n";
                    err.clear();
                }
            }
        }
        if (key == '1') selected = 0;
        if (key == '2') selected = 1;
        if (key == '[') move_step = std::max(0.001, move_step * 0.5);
        if (key == ']') move_step = std::min(0.25, move_step * 2.0);
        if (key == ',') size_step = std::max(0.001, size_step * 0.5);
        if (key == '.') size_step = std::min(0.25, size_step * 2.0);
        if (key == 'r') {
            rois = opt.default_rois;
            std::cout << "\n[roi] reset to defaults\n";
        }
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
                            "- Tag fill ratio: " + std::to_string(locked_pack.tag_fill_ratio) + "\n" +
                            "- red_roi ratio: " + std::to_string(rois.red_roi.x) + "," + std::to_string(rois.red_roi.y) + "," + std::to_string(rois.red_roi.w) + "," + std::to_string(rois.red_roi.h) + "\n" +
                            "- image_roi ratio: " + std::to_string(rois.image_roi.x) + "," + std::to_string(rois.image_roi.y) + "," + std::to_string(rois.image_roi.w) + "," + std::to_string(rois.image_roi.h));
            std::cout << "\n[save] all -> " << opt.save_warp << " ; " << opt.save_rois << " ; " << opt.save_report << "\n";
        }
    }

    release_model_runtime();
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
    if (opt.model_cfg.enable && opt.model_cfg.backend != "off" && !opt.model_cfg.path.empty()) {
        std::string merr;
        model_ready = init_model_runtime(opt.model_cfg, merr);
        if (!model_ready) std::cerr << "\n[model] disabled: " << merr << "\n";
    }

    int cam_w = opt.width, cam_h = opt.height;
    cv::VideoCapture cap;
    if (!open_capture(cap, opt.device, cam_w, cam_h, opt.fps, opt.fourcc, opt.buffer_size, opt.camera_soft_max, err)) return false;

    cv::Mat frame, warped, valid, camera_show, warp_show;
    RoiRuntimeData roi_info;
    ModelResult model_res;
    cv::namedWindow("vision_app_camera", cv::WINDOW_NORMAL);
    cv::namedWindow("vision_app_warp", cv::WINDOW_NORMAL);
    cv::resizeWindow("vision_app_camera", opt.camera_preview_max, std::max(240, opt.camera_preview_max * 3 / 4));
    cv::resizeWindow("vision_app_warp", opt.warp_preview_max, opt.warp_preview_max);
    std::cout << "\n=== vision_app deploy ===\n"
              << "Build every time with:\n"
              << "  mkdir -p build\n"
              << "  cd build\n"
              << "  cmake ..\n"
              << "  make -j$(nproc)\n\n"
              << "q / ESC quit\n"
              << "loaded warp: " << opt.load_warp << "\n"
              << "loaded rois: " << opt.load_rois << "\n";

    int frame_idx = 0;
    while (true) {
        if (!grab_latest_frame(cap, opt.latest_only, opt.drain_grabs, frame)) { err = "failed to read frame"; break; }
        if (!apply_warp(frame, pack, warped, &valid)) { err = "failed to apply warp"; break; }
        camera_show = downscale_for_preview(frame, opt.camera_preview_max);
        warp_show = warped.clone();
        draw_rois(warp_show, rois, -1);
        cv::putText(warp_show, "DEPLOY family=" + pack.family + " id=" + std::to_string(pack.id), {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.68, cv::Scalar(0,120,0), 2);
        std::string rerr;
        if (extract_runtime_rois(warped, valid, rois, opt.red_cfg, roi_info, rerr)) {
            cv::putText(warp_show, "red_ratio=" + std::to_string(roi_info.red_ratio).substr(0,5), {12, 56}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0,120,0), 2);
            if (model_ready && (frame_idx % std::max(1, opt.model_cfg.stride) == 0)) {
                std::string merr;
                if (run_model_on_image_roi(roi_info, opt.model_cfg, model_res, merr)) {
                    cv::putText(warp_show, model_res.summary, {12, 80}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,120,0), 1);
                }
            }
        }
        warp_show = downscale_for_preview(warp_show, opt.warp_preview_max);
        cv::imshow("vision_app_camera", camera_show);
        cv::imshow("vision_app_warp", warp_show);
        print_status_to_terminal(true, AprilTagDetection{true, pack.family, pack.id}, 1, 0.0, 0.0, pack.tag_fill_ratio, &roi_info, &model_res);
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
