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

struct DeployConfig {
    std::string mode = "live"; // probe | bench | live | deploy

    std::string save_probe_csv = "../report/probe_table.csv";
    std::string save_test_csv = "../report/test_results.csv";
    std::string save_report_md = "../report/latest_report.md";
    std::string save_h_path = "../report/warp_h.json";
    std::string save_rois_path = "../report/rois.json";
    std::string load_h_path = "";
    std::string load_rois_path = "";

    bool auto_save_lock = true;
    bool auto_load_h = false;
    bool auto_load_rois = false;

    int warp_width = 1280;
    int warp_height = 720;

    bool live_preview_raw = true;
    bool live_preview_warp = true;
    bool save_snapshots = false;
    std::string snapshot_dir = "../report";
};

static void nudge_ratio(RoiRatio& roi, char key, double move_step, double size_step) {
    if (key == 'a') roi.x -= move_step;
    if (key == 'd') roi.x += move_step;
    if (key == 'w') roi.y -= move_step;
    if (key == 's') roi.y += move_step;
    if (key == 'j') roi.w -= size_step;
    if (key == 'l') roi.w += size_step;
    if (key == 'i') roi.h -= size_step;
    if (key == 'k') roi.h += size_step;
    clamp_and_validate_roi(roi);
}

static void print_live_keys() {
    std::cout << "\nLive mode keys:\n"
              << "  q / ESC : quit\n"
              << "  p       : save homography + rois\n"
              << "  u       : unlock / reacquire tag\n"
              << "  1       : select red_roi\n"
              << "  2       : select image_roi\n"
              << "  w a s d : move selected roi\n"
              << "  i j k l : resize selected roi (h/w)\n"
              << "  r       : reset rois to defaults\n"
              << "  c       : capture warped snapshot\n\n";
}

static bool run_bench_mode(const CameraConfig& cam_cfg,
                           const DeployConfig& dep_cfg,
                           CaptureStats& out_capture,
                           std::string& err) {
    LatestFrameCamera cam;
    if (!cam.open(cam_cfg, err)) return false;

    cv::Mat frame;
    const auto t0 = std::chrono::steady_clock::now();
    while (true) {
        if (!cam.read_latest(frame, err)) return false;
        if (!cam_cfg.headless && cam_cfg.preview) {
            cv::imshow("vision_app_bench", frame);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) break;
        }
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (elapsed >= cam_cfg.duration_sec) break;
    }
    out_capture = cam.stats();
    cam.close();
    if (!cam_cfg.headless && cam_cfg.preview) cv::destroyWindow("vision_app_bench");

    print_capture_stats(cam_cfg, out_capture);
    append_test_csv(dep_cfg.save_test_csv, cam_cfg, out_capture);
    write_latest_report_md(dep_cfg.save_report_md, cam_cfg, out_capture, nullptr, nullptr, nullptr);
    return true;
}

static bool run_live_or_deploy_mode(const CameraConfig& cam_cfg,
                                    const AprilTagConfig& tag_cfg,
                                    const DeployConfig& dep_cfg,
                                    RoiSet& rois,
                                    CaptureStats& out_capture,
                                    StageStats& out_stage,
                                    HomographyLock& out_lock,
                                    std::string& err) {
    LatestFrameCamera cam;
    if (!cam.open(cam_cfg, err)) return false;

    if (dep_cfg.auto_load_rois || !dep_cfg.load_rois_path.empty()) {
        const std::string path = dep_cfg.load_rois_path.empty() ? dep_cfg.save_rois_path : dep_cfg.load_rois_path;
        load_rois_json(path, rois);
    }

    HomographyLock lock;
    if (dep_cfg.mode == "deploy" || dep_cfg.auto_load_h || !dep_cfg.load_h_path.empty()) {
        const std::string path = dep_cfg.load_h_path.empty() ? dep_cfg.save_h_path : dep_cfg.load_h_path;
        if (!load_homography_json(path, lock)) {
            if (dep_cfg.mode == "deploy") {
                err = "deploy mode requires a valid saved homography file";
                return false;
            }
        }
    }

    TagLocker locker(tag_cfg);
    RollingTimer detect_t, warp_t, total_t;
    cv::Mat frame, raw_view, warped, red_crop, image_crop;
    char selected_roi = '1';
    print_live_keys();

    const auto t0 = std::chrono::steady_clock::now();
    while (true) {
        const auto loop_start = std::chrono::steady_clock::now();
        if (!cam.read_latest(frame, err)) return false;

        raw_view = frame.clone();
        AprilTagDetection det;
        bool have_det = false;

        if (!lock.valid) {
            const auto td0 = std::chrono::steady_clock::now();
            if (!detect_apriltag_best(frame, tag_cfg, det, err)) return false;
            const double detect_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
            detect_t.push(detect_ms);
            have_det = det.found;

            const bool got_lock = have_det && locker.update(det, frame.size());
            draw_detection_overlay(raw_view, det, got_lock);
            if (got_lock) {
                if (!compute_homography_from_tag_quad(locker.locked_det(), cv::Size(dep_cfg.warp_width, dep_cfg.warp_height), lock)) {
                    err = "failed to compute homography after tag lock";
                    return false;
                }
                if (dep_cfg.auto_save_lock) {
                    save_homography_json(dep_cfg.save_h_path, lock);
                    save_rois_json(dep_cfg.save_rois_path, rois);
                }
            }
        } else {
            cv::putText(raw_view, "Locked deploy mode", {20, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,255,0}, 2);
        }

        if (lock.valid) {
            const auto tw0 = std::chrono::steady_clock::now();
            if (!warp_full_frame(frame, warped, lock)) {
                err = "warpPerspective failed";
                return false;
            }
            const double warp_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw0).count();
            warp_t.push(warp_ms);
            draw_roi_overlay(warped, rois, lock);
            red_crop = crop_roi_clone(warped, rois.red_roi);
            image_crop = crop_roi_clone(warped, rois.image_roi);
            if (!red_crop.empty()) cv::imshow("vision_app_red_roi", red_crop);
            if (!image_crop.empty()) cv::imshow("vision_app_image_roi", image_crop);
        }

        if (!cam_cfg.headless && dep_cfg.live_preview_raw) cv::imshow("vision_app_raw", raw_view);
        if (!cam_cfg.headless && dep_cfg.live_preview_warp && !warped.empty()) cv::imshow("vision_app_warp", warped);

        const double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - loop_start).count();
        total_t.push(total_ms);

        const int key = (cam_cfg.headless ? -1 : (cv::waitKey(1) & 0xFF));
        if (key == 'q' || key == 27) break;
        if (key == '1') selected_roi = '1';
        if (key == '2') selected_roi = '2';
        if (key == 'u') { lock = {}; locker.reset(); warped.release(); }
        if (key == 'r') { rois = {}; }
        if (key == 'p') {
            if (lock.valid) save_homography_json(dep_cfg.save_h_path, lock);
            save_rois_json(dep_cfg.save_rois_path, rois);
            std::cout << "Saved: " << dep_cfg.save_h_path << " and " << dep_cfg.save_rois_path << "\n";
        }
        if (key == 'c' && dep_cfg.save_snapshots && !warped.empty()) {
            const std::string path = dep_cfg.snapshot_dir + "/warped_snapshot.png";
            cv::imwrite(path, warped);
            std::cout << "Saved snapshot: " << path << "\n";
        }

        if (selected_roi == '1') nudge_ratio(rois.red_roi, static_cast<char>(key), 0.005, 0.005);
        else if (selected_roi == '2') nudge_ratio(rois.image_roi, static_cast<char>(key), 0.005, 0.005);

        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (cam_cfg.duration_sec > 0 && elapsed >= cam_cfg.duration_sec && cam_cfg.headless) break;
    }

    out_capture = cam.stats();
    out_stage.frames = out_capture.frames;
    out_stage.detect_ms_avg = detect_t.avg();
    out_stage.warp_ms_avg = warp_t.avg();
    out_stage.total_ms_avg = total_t.avg();
    out_stage.detect_ms_max = detect_t.max_ms;
    out_stage.warp_ms_max = warp_t.max_ms;
    out_stage.total_ms_max = total_t.max_ms;
    out_stage.detector_fps_avg = (out_stage.detect_ms_avg > 0.0) ? (1000.0 / out_stage.detect_ms_avg) : 0.0;
    out_lock = lock;

    cam.close();
    cv::destroyAllWindows();

    print_capture_stats(cam_cfg, out_capture);
    append_test_csv(dep_cfg.save_test_csv, cam_cfg, out_capture, &out_stage, lock.valid ? &out_lock : nullptr);
    write_latest_report_md(dep_cfg.save_report_md, cam_cfg, out_capture, &out_stage, lock.valid ? &out_lock : nullptr, &rois);
    if (lock.valid) {
        save_homography_json(dep_cfg.save_h_path, lock);
        save_rois_json(dep_cfg.save_rois_path, rois);
    }
    return true;
}

} // namespace vision_app
