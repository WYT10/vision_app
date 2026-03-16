#pragma once

#include <chrono>
#include <iostream>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "camera.hpp"
#include "calibrate.hpp"
#include "stats.hpp"

namespace vision_app {

struct AppOptions {
    std::string mode = "live";
    CameraOptions cam{};
    AprilTagConfig tag{};
    int duration = 10;
    int warp_soft_max = 900;
    int preview_soft_max = 600;
    std::string save_warp = "../report/warp_package.yml.gz";
    std::string save_rois = "../report/rois.yml";
    std::string load_warp = "../report/warp_package.yml.gz";
    std::string load_rois = "../report/rois.yml";
    std::string csv_path = "../report/test_results.csv";
    std::string md_path = "../report/latest_report.md";
    bool show_help = true;
};

static inline cv::Mat downscale_for_preview(const cv::Mat& img, int soft_max) {
    if (img.empty()) return img;
    if (soft_max <= 0) return img;
    int w = img.cols, h = img.rows;
    int m = std::max(w, h);
    if (m <= soft_max) return img;
    double scale = static_cast<double>(soft_max) / static_cast<double>(m);
    cv::Mat out;
    cv::resize(img, out, cv::Size(std::max(1, int(std::round(w * scale))), std::max(1, int(std::round(h * scale)))), 0, 0, cv::INTER_AREA);
    return out;
}

static inline void draw_help_overlay(cv::Mat& img, int selected_roi, double move_step, double size_step) {
    int y = 24;
    auto put = [&](const std::string& s) {
        cv::putText(img, s, {16, y}, cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(255,255,255), 1);
        y += 20;
    };
    put("space lock | u unlock | 1 red_roi | 2 image_roi | p save all");
    put("wasd move | j/l width -/+ | i/k height -/+ | [ ] move step | , . size step");
    put("r reset selected | y save warp | o save rois | h help | q quit");
    put(std::string("selected=") + (selected_roi == 1 ? "red_roi" : "image_roi") +
        " move_step=" + cv::format("%.4f", move_step) +
        " size_step=" + cv::format("%.4f", size_step));
}

static inline bool run_bench(const AppOptions& opt, std::string& err) {
    CameraCapture cam;
    if (!cam.open(opt.cam, err)) return false;
    cv::Mat frame;
    for (int i = 0; i < 5; ++i) cam.read_latest(frame);

    BenchStats st{};
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    auto prev = t0;
    double dt_sum = 0.0, dt_min = 1e18, dt_max = 0.0;

    while (true) {
        if (!cam.read_latest(frame)) { err = "camera read failed"; return false; }
        auto now = clk::now();
        double dt_ms = std::chrono::duration<double, std::milli>(now - prev).count();
        prev = now;
        if (st.frames > 0) {
            dt_sum += dt_ms;
            dt_min = std::min(dt_min, dt_ms);
            dt_max = std::max(dt_max, dt_ms);
        }
        st.frames++;
        st.actual_w = frame.cols; st.actual_h = frame.rows;
        if (!opt.cam.headless) {
            cv::Mat disp = downscale_for_preview(frame, opt.preview_soft_max);
            cv::imshow("vision_app", disp);
            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) break;
        }
        double elapsed = std::chrono::duration<double>(now - t0).count();
        if (elapsed >= opt.duration) break;
    }
    st.elapsed_sec = std::chrono::duration<double>(clk::now() - t0).count();
    if (st.elapsed_sec > 0.0) st.fps_avg = static_cast<double>(st.frames) / st.elapsed_sec;
    if (st.frames > 1) {
        st.frame_ms_avg = dt_sum / static_cast<double>(st.frames - 1);
        st.frame_ms_min = dt_min;
        st.frame_ms_max = dt_max;
        if (dt_max > 0.0) st.fps_min = 1000.0 / dt_max;
        if (dt_min > 0.0) st.fps_max = 1000.0 / dt_min;
    }
    print_bench_stats(st);
    append_bench_csv(opt.csv_path, opt.cam.device, opt.cam.width, opt.cam.height, opt.cam.fps, opt.cam.fourcc, st);
    write_latest_report_md(opt.md_path, "bench",
        "- Device: " + opt.cam.device + "\n" +
        "- Requested: " + std::to_string(opt.cam.width) + "x" + std::to_string(opt.cam.height) + " @ " + std::to_string(opt.cam.fps) + " " + opt.cam.fourcc + "\n" +
        "- Actual FPS: " + cv::format("%.3f", st.fps_avg) + "\n");
    return true;
}

static inline bool run_live(const AppOptions& opt, std::string& err) {
#if !VISION_APP_HAS_ARUCO
    err = "OpenCV ArUco/AprilTag support not available in this build";
    return false;
#else
    CameraCapture cam;
    if (!cam.open(opt.cam, err)) return false;
    RoiConfig rois;
    load_rois_yaml(opt.save_rois, rois);
    WarpPackage locked_pack;
    TagLocker locker(opt.tag.lock_frames);
    AprilTagDetection det, locked_det;
    cv::Mat frame, warped, valid, display;
    bool locked = false;
    int selected = 1;
    double move_step = 0.005;
    double size_step = 0.005;
    bool show_help = opt.show_help;
    cv::namedWindow("vision_app", cv::WINDOW_AUTOSIZE);

    while (true) {
        if (!cam.read_latest(frame)) { err = "camera read failed"; return false; }
        AprilTagDetection cur{};
        detect_apriltag_best(frame, opt.tag, cur, err);
        if (!locked) {
            if (cur.found && !opt.tag.manual_lock_only && locker.update(cur)) {
                if (!build_warp_package_from_detection(cur, frame.size(), opt.warp_soft_max, locked_pack, err)) return false;
                locked = true; locked_det = cur;
            }
        }

        if (!locked) {
            cv::Mat raw = frame.clone();
            draw_detection(raw, cur, false);
            if (cur.found) {
                WarpPackage temp_pack;
                if (build_warp_package_from_detection(cur, frame.size(), opt.warp_soft_max, temp_pack, err)) {
                    apply_warp(frame, temp_pack, warped, valid);
                    cv::Mat warp_preview = compose_preview_with_mask(warped, valid);
                    draw_rois(warp_preview, rois, selected);
                    cv::Mat left = downscale_for_preview(raw, opt.preview_soft_max);
                    cv::Mat right = downscale_for_preview(warp_preview, opt.preview_soft_max);
                    int H = std::max(left.rows, right.rows);
                    int W = left.cols + right.cols;
                    display = cv::Mat(H, W, CV_8UC3, cv::Scalar(30,30,30));
                    left.copyTo(display(cv::Rect(0,0,left.cols,left.rows)));
                    right.copyTo(display(cv::Rect(left.cols,0,right.cols,right.rows)));
                } else {
                    display = downscale_for_preview(raw, opt.preview_soft_max);
                }
            } else {
                display = downscale_for_preview(raw, opt.preview_soft_max);
            }
            if (show_help) draw_help_overlay(display, selected, move_step, size_step);
            cv::imshow("vision_app", display);
        } else {
            if (!apply_warp(frame, locked_pack, warped, valid)) { err = "apply_warp failed"; return false; }
            cv::Mat warp_preview = compose_preview_with_mask(warped, valid);
            draw_rois(warp_preview, rois, selected);
            cv::putText(warp_preview, "LOCKED family=" + locked_pack.family + " id=" + std::to_string(locked_pack.id),
                        {16, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0,255,0), 2);
            display = downscale_for_preview(warp_preview, opt.preview_soft_max);
            if (show_help) draw_help_overlay(display, selected, move_step, size_step);
            cv::imshow("vision_app", display);
        }

        int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') break;
        if (key == 'h') show_help = !show_help;
        if (key == '1') selected = 1;
        if (key == '2') selected = 2;
        if (key == '[') move_step = std::max(0.001, move_step * 0.5);
        if (key == ']') move_step = std::min(0.05, move_step * 2.0);
        if (key == ',') size_step = std::max(0.001, size_step * 0.5);
        if (key == '.') size_step = std::min(0.10, size_step * 2.0);
        if (key == 'r') {
            if (selected == 1) rois.red_roi = RoiRatio{0.10,0.10,0.20,0.20};
            else rois.image_roi = RoiRatio{0.35,0.15,0.45,0.45};
        }
        if (key == 'u') { locked = false; locker.reset(); locked_pack = {}; }
        if (key == ' ' || key == 13) {
            if (cur.found) {
                if (!build_warp_package_from_detection(cur, frame.size(), opt.warp_soft_max, locked_pack, err)) return false;
                locked = true; locked_det = cur;
            }
        }
        if (key == 'p') {
            if (locked_pack.valid && !save_warp_package(opt.save_warp, locked_pack)) std::cerr << "Warning: failed to save warp package\n";
            if (!save_rois_yaml(opt.save_rois, rois)) std::cerr << "Warning: failed to save rois\n";
            write_latest_report_md(opt.md_path, "live",
                "- Locked family: " + locked_pack.family + "\n" +
                "- Locked id: " + std::to_string(locked_pack.id) + "\n" +
                "- Warp size: " + std::to_string(locked_pack.warp_w) + "x" + std::to_string(locked_pack.warp_h) + "\n" +
                "- Save warp: `" + opt.save_warp + "`\n" +
                "- Save rois: `" + opt.save_rois + "`\n");
        }
        if (key == 'y' && locked_pack.valid) {
            if (!save_warp_package(opt.save_warp, locked_pack)) std::cerr << "Warning: failed to save warp package\n";
        }
        if (key == 'o') {
            if (!save_rois_yaml(opt.save_rois, rois)) std::cerr << "Warning: failed to save rois\n";
        }

        if (selected == 1) nudge_roi(rois.red_roi, static_cast<char>(key), move_step, size_step);
        else nudge_roi(rois.image_roi, static_cast<char>(key), move_step, size_step);
    }
    cv::destroyWindow("vision_app");
    return true;
#endif
}

static inline bool run_deploy(const AppOptions& opt, std::string& err) {
    CameraCapture cam;
    if (!cam.open(opt.cam, err)) return false;
    WarpPackage pack;
    if (!load_warp_package(opt.load_warp, pack)) {
        err = "failed to load warp package: " + opt.load_warp;
        return false;
    }
    RoiConfig rois;
    load_rois_yaml(opt.load_rois, rois);
    cv::namedWindow("vision_app", cv::WINDOW_AUTOSIZE);
    cv::Mat frame, warped, valid, display;
    while (true) {
        if (!cam.read_latest(frame)) { err = "camera read failed"; return false; }
        if (!apply_warp(frame, pack, warped, valid)) { err = "apply_warp failed"; return false; }
        display = compose_preview_with_mask(warped, valid);
        draw_rois(display, rois, 0);
        cv::Rect rr = roi_to_rect(rois.red_roi, display.size());
        cv::Rect ir = roi_to_rect(rois.image_roi, display.size());
        double valid_ratio_red = static_cast<double>(cv::countNonZero(valid(rr))) / static_cast<double>(rr.area());
        double valid_ratio_img = static_cast<double>(cv::countNonZero(valid(ir))) / static_cast<double>(ir.area());
        cv::putText(display, "deploy red_valid=" + cv::format("%.2f", valid_ratio_red) + " image_valid=" + cv::format("%.2f", valid_ratio_img),
                    {16,24}, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 2);
        cv::Mat preview = downscale_for_preview(display, opt.preview_soft_max);
        cv::imshow("vision_app", preview);
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) break;
    }
    cv::destroyWindow("vision_app");
    return true;
}

static inline void print_mode_help(const std::string& mode) {
    if (mode == "bench") {
        std::cout << "bench: open one camera mode and measure real runtime FPS.\n"
                  << "Important args: --device --width --height --fps --fourcc --duration --buffer-size --latest-only --drain-grabs\n";
    } else if (mode == "live") {
        std::cout << "live: detect AprilTag, preview temporary full-image auto-fit warp, lock, edit ROIs, save precomputation.\n"
                  << "Important args: --tag-family auto|16|25|36 --target-id --require-target-id 0|1 --manual-lock-only 0|1 --lock-frames N --warp-soft-max N --preview-soft-max N\n"
                  << "Soft limit behavior: if the fitted warp would exceed --warp-soft-max, it is scaled down automatically. A hard safety cap of 1200 px max dimension is also applied.\n";
    } else if (mode == "deploy") {
        std::cout << "deploy: load saved warp package + ROI config and run directly with remap cache.\n"
                  << "Important args: --load-warp PATH --load-rois PATH --preview-soft-max N\n";
    } else {
        std::cout << "probe: list camera-reported formats / resolutions / fps.\n";
    }
}

} // namespace vision_app
