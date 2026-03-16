#pragma once

#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
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
    std::string save_remap_path = "../report/warp_remap.yml.gz";
    std::string load_h_path = "";
    std::string load_rois_path = "";
    std::string load_remap_path = "";

    bool auto_save_lock = true;
    bool auto_load_h = false;
    bool auto_load_rois = false;
    bool auto_load_remap = false;

    int warp_width = 1280;
    int warp_height = 720;

    bool live_preview_raw = true;
    bool live_preview_warp = true;
    bool show_roi_crops = false;
    bool show_help_overlay = true;
    bool show_status_overlay = true;
    bool save_snapshots = false;
    std::string snapshot_dir = "../report";

    bool use_remap_cache = true;
    bool fixed_point_remap = true;
    bool save_remap_cache = true;

    double move_step = 0.005;
    double size_step = 0.005;
};

static RoiRatio default_red_roi() { return RoiRatio{0.05, 0.10, 0.20, 0.20}; }
static RoiRatio default_image_roi() { return RoiRatio{0.30, 0.10, 0.50, 0.60}; }

static void nudge_ratio(RoiRatio& roi, int key, double move_step, double size_step) {
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
    std::cout
        << "\nLive mode keys (keyboard-only ROI flow):\n"
        << "  q / ESC      quit\n"
        << "  h            toggle help overlay\n"
        << "  space/enter  lock current visible tag\n"
        << "  u            unlock and reacquire\n"
        << "  p            save all outputs\n"
        << "  y            save homography only\n"
        << "  o            save rois only\n"
        << "  g            save remap cache only\n"
        << "  1 / 2        select red_roi / image_roi\n"
        << "  TAB          switch selected roi\n"
        << "  wasd         move selected roi\n"
        << "  ijkl         resize selected roi (height/width)\n"
        << "  [ / ]        adjust move step\n"
        << "  , / .        adjust size step\n"
        << "  r            reset both rois to defaults\n"
        << "  c            save warped snapshot\n"
        << std::endl;
}

static std::string live_phase_string(bool lock_valid, const TagLocker& locker, bool manual_lock_only) {
    if (lock_valid) return "LOCKED";
    if (manual_lock_only) return "SEARCHING (manual lock)";
    if (locker.history_size() > 0) return "LOCKING";
    return "SEARCHING";
}

static void draw_live_hud(cv::Mat& raw_view,
                          cv::Mat* warped_view,
                          const AprilTagConfig& tag_cfg,
                          const TagLocker& locker,
                          const AprilTagDetection& det,
                          const HomographyLock& lock,
                          const WarpRemapCache* remap_cache,
                          char selected_roi,
                          bool show_help_overlay,
                          bool auto_save_lock,
                          bool use_remap_cache,
                          double move_step,
                          double size_step) {
    const std::string phase = live_phase_string(lock.valid, locker, tag_cfg.manual_lock_only);
    std::ostringstream line1;
    line1 << "phase=" << phase << "  family_req=" << normalize_family_token(tag_cfg.family)
          << "  target_id=" << (tag_cfg.require_target_id ? std::to_string(tag_cfg.target_id) : std::string("any"));
    cv::putText(raw_view, line1.str(), {20, 60}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {255,255,255}, 2);

    std::ostringstream line2;
    line2 << "selected_roi=" << (selected_roi == '1' ? "red_roi" : "image_roi")
          << "  move_step=" << std::fixed << std::setprecision(4) << move_step
          << "  size_step=" << size_step
          << "  auto_save=" << (auto_save_lock ? "on" : "off");
    cv::putText(raw_view, line2.str(), {20, 84}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {255,255,255}, 2);

    if (det.found && !lock.valid) {
        std::ostringstream line3;
        line3 << "candidate=" << det.family << " id=" << det.id << "  hist=" << locker.history_size() << '/' << tag_cfg.lock_frames;
        cv::putText(raw_view, line3.str(), {20, 108}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {0,255,255}, 2);
    }

    if (show_help_overlay) {
        const std::vector<std::string> lines = {
            "space lock  u unlock  p save all  y save H  o save rois  g save remap",
            "1/2 select roi  wasd move  ijkl resize  [ ] move step  , . size step",
            std::string("warp=") + (use_remap_cache ? "remap cache" : "warpPerspective") +
                ((remap_cache && remap_cache->valid) ? " [ready]" : " [not ready]"),
            std::string("family=") + normalize_family_token(tag_cfg.family) + "  options: auto|16|25|36"
        };
        int y = raw_view.rows - 60;
        for (const auto& line : lines) {
            cv::putText(raw_view, line, {20, y}, cv::FONT_HERSHEY_SIMPLEX, 0.48, {220,220,220}, 1);
            y += 18;
        }
    }

    if (warped_view && !warped_view->empty()) {
        std::ostringstream w1;
        w1 << (selected_roi == '1' ? "red_roi" : "image_roi") << "  move="
           << std::fixed << std::setprecision(4) << move_step << " size=" << size_step;
        cv::putText(*warped_view, w1.str(), {20, warped_view->rows - 22}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {255,255,255}, 2);
    }
}

static bool save_all_outputs(const DeployConfig& dep_cfg,
                             const CameraConfig& cam_cfg,
                             const CaptureStats& capture,
                             const StageStats& stage,
                             const HomographyLock* lock,
                             const RoiSet& rois,
                             const WarpRemapCache* remap_cache) {
    bool ok = true;
    if (lock && lock->valid) ok = save_homography_json(dep_cfg.save_h_path, *lock) && ok;
    if (dep_cfg.save_remap_cache && remap_cache && remap_cache->valid) {
        ok = save_warp_remap_cache(dep_cfg.save_remap_path, *remap_cache) && ok;
    }
    ok = save_rois_json(dep_cfg.save_rois_path, rois) && ok;
    ok = write_latest_report_md(dep_cfg.save_report_md, cam_cfg, capture, &stage, (lock && lock->valid) ? lock : nullptr, &rois) && ok;
    return ok;
}

static bool ensure_remap_cache_ready(const DeployConfig& dep_cfg,
                                     const cv::Size& source_size,
                                     const HomographyLock& lock,
                                     WarpRemapCache& cache,
                                     std::string& err) {
    if (!dep_cfg.use_remap_cache || !lock.valid) return true;
    if (remap_cache_matches(cache, source_size, lock.warp_size)) return true;

    if ((dep_cfg.mode == "deploy" || dep_cfg.auto_load_remap || !dep_cfg.load_remap_path.empty()) && !cache.valid) {
        const std::string path = dep_cfg.load_remap_path.empty() ? dep_cfg.save_remap_path : dep_cfg.load_remap_path;
        WarpRemapCache loaded;
        if (load_warp_remap_cache(path, loaded) && remap_cache_matches(loaded, source_size, lock.warp_size)) {
            cache = loaded;
            return true;
        }
    }

    if (!build_warp_remap_cache(lock, source_size, cache, dep_cfg.fixed_point_remap)) {
        err = "failed to build warp remap cache";
        return false;
    }
    if (dep_cfg.save_remap_cache) {
        save_warp_remap_cache(dep_cfg.save_remap_path, cache);
    }
    return true;
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
    cv::Mat frame, raw_view, warped;
    WarpRemapCache remap_cache;
    char selected_roi = '1';
    bool show_help_overlay = dep_cfg.show_help_overlay;
    bool auto_save_lock = dep_cfg.auto_save_lock;
    double move_step = dep_cfg.move_step;
    double size_step = dep_cfg.size_step;
    print_live_keys();

    if (!cam_cfg.headless) {
        if (dep_cfg.live_preview_raw) cv::namedWindow("vision_app_raw", cv::WINDOW_AUTOSIZE);
        if (dep_cfg.live_preview_warp) cv::namedWindow("vision_app_warp", cv::WINDOW_AUTOSIZE);
        if (dep_cfg.show_roi_crops) {
            cv::namedWindow("vision_app_red_roi", cv::WINDOW_AUTOSIZE);
            cv::namedWindow("vision_app_image_roi", cv::WINDOW_AUTOSIZE);
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    while (true) {
        const auto loop_start = std::chrono::steady_clock::now();
        if (!cam.read_latest(frame, err)) return false;

        raw_view = frame.clone();
        AprilTagDetection det;

        if (!lock.valid) {
            const auto td0 = std::chrono::steady_clock::now();
            if (!detect_apriltag_best(frame, tag_cfg, det, err)) return false;
            const double detect_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
            detect_t.push(detect_ms);

            const bool allow_auto_lock = !tag_cfg.manual_lock_only;
            const bool got_lock = det.found && locker.update(det, frame.size(), allow_auto_lock);
            draw_detection_overlay(raw_view, det, got_lock);
            if (got_lock) {
                if (!compute_homography_from_tag_quad(locker.locked_det(), cv::Size(dep_cfg.warp_width, dep_cfg.warp_height), lock)) {
                    err = "failed to compute homography after tag lock";
                    return false;
                }
                if (!ensure_remap_cache_ready(dep_cfg, frame.size(), lock, remap_cache, err)) return false;
                if (auto_save_lock) {
                    save_homography_json(dep_cfg.save_h_path, lock);
                    save_rois_json(dep_cfg.save_rois_path, rois);
                    if (dep_cfg.save_remap_cache && remap_cache.valid) save_warp_remap_cache(dep_cfg.save_remap_path, remap_cache);
                }
            }
        } else {
            cv::putText(raw_view, "Locked deploy/runtime mode", {20, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,255,0}, 2);
        }

        if (lock.valid) {
            if (!ensure_remap_cache_ready(dep_cfg, frame.size(), lock, remap_cache, err)) return false;
            const auto tw0 = std::chrono::steady_clock::now();
            const bool warped_ok = dep_cfg.use_remap_cache
                ? warp_full_frame_cached(frame, warped, remap_cache)
                : warp_full_frame(frame, warped, lock);
            if (!warped_ok) {
                err = dep_cfg.use_remap_cache ? "remap warp failed" : "warpPerspective failed";
                return false;
            }
            const double warp_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw0).count();
            warp_t.push(warp_ms);
            draw_roi_overlay(warped, rois, lock, selected_roi);
            draw_live_hud(raw_view, &warped, tag_cfg, locker, det, lock, &remap_cache, selected_roi, show_help_overlay, auto_save_lock, dep_cfg.use_remap_cache, move_step, size_step);
            if (!cam_cfg.headless && dep_cfg.show_roi_crops) {
                const cv::Mat red_crop = crop_roi_clone(warped, rois.red_roi);
                const cv::Mat image_crop = crop_roi_clone(warped, rois.image_roi);
                if (!red_crop.empty()) cv::imshow("vision_app_red_roi", red_crop);
                if (!image_crop.empty()) cv::imshow("vision_app_image_roi", image_crop);
            }
        } else {
            draw_live_hud(raw_view, nullptr, tag_cfg, locker, det, lock, nullptr, selected_roi, show_help_overlay, auto_save_lock, dep_cfg.use_remap_cache, move_step, size_step);
        }

        if (!cam_cfg.headless && dep_cfg.live_preview_raw) cv::imshow("vision_app_raw", raw_view);
        if (!cam_cfg.headless && dep_cfg.live_preview_warp && !warped.empty()) cv::imshow("vision_app_warp", warped);

        const double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - loop_start).count();
        total_t.push(total_ms);

        const int key = (cam_cfg.headless ? -1 : (cv::waitKey(1) & 0xFF));
        if (key == 'q' || key == 27) break;
        if (key == '1') selected_roi = '1';
        if (key == '2') selected_roi = '2';
        if (key == 9) selected_roi = (selected_roi == '1') ? '2' : '1';
        if (key == 'h') show_help_overlay = !show_help_overlay;
        if (key == 'u') { lock = {}; locker.reset(); warped.release(); remap_cache = {}; }
        if (key == 'r') { rois.red_roi = default_red_roi(); rois.image_roi = default_image_roi(); }
        if (key == 'p') {
            StageStats stage_preview{};
            stage_preview.frames = cam.stats().frames;
            stage_preview.detect_ms_avg = detect_t.avg();
            stage_preview.warp_ms_avg = warp_t.avg();
            stage_preview.total_ms_avg = total_t.avg();
            stage_preview.detect_ms_max = detect_t.max_ms;
            stage_preview.warp_ms_max = warp_t.max_ms;
            stage_preview.total_ms_max = total_t.max_ms;
            stage_preview.detector_fps_avg = (stage_preview.detect_ms_avg > 0.0) ? (1000.0 / stage_preview.detect_ms_avg) : 0.0;
            save_all_outputs(dep_cfg, cam_cfg, cam.stats(), stage_preview, lock.valid ? &lock : nullptr, rois, remap_cache.valid ? &remap_cache : nullptr);
            std::cout << "Saved all outputs\n";
        }
        if (key == 'y' && lock.valid) {
            save_homography_json(dep_cfg.save_h_path, lock);
            std::cout << "Saved homography: " << dep_cfg.save_h_path << "\n";
        }
        if (key == 'o') {
            save_rois_json(dep_cfg.save_rois_path, rois);
            std::cout << "Saved rois: " << dep_cfg.save_rois_path << "\n";
        }
        if (key == 'g' && remap_cache.valid) {
            save_warp_remap_cache(dep_cfg.save_remap_path, remap_cache);
            std::cout << "Saved remap cache: " << dep_cfg.save_remap_path << "\n";
        }
        if ((key == ' ' || key == 13) && !lock.valid) {
            const AprilTagDetection cand = locker.last_candidate().found ? locker.last_candidate() : det;
            if (locker.force_lock(cand, frame.size())) {
                if (!compute_homography_from_tag_quad(locker.locked_det(), cv::Size(dep_cfg.warp_width, dep_cfg.warp_height), lock)) {
                    err = "failed to compute homography after manual lock";
                    return false;
                }
                if (!ensure_remap_cache_ready(dep_cfg, frame.size(), lock, remap_cache, err)) return false;
                if (auto_save_lock) {
                    save_homography_json(dep_cfg.save_h_path, lock);
                    save_rois_json(dep_cfg.save_rois_path, rois);
                    if (dep_cfg.save_remap_cache && remap_cache.valid) save_warp_remap_cache(dep_cfg.save_remap_path, remap_cache);
                }
            }
        }
        if (key == 'c' && dep_cfg.save_snapshots && !warped.empty()) {
            const std::string path = dep_cfg.snapshot_dir + "/warped_snapshot.png";
            cv::imwrite(path, warped);
            std::cout << "Saved snapshot: " << path << "\n";
        }
        if (key == '[') move_step = std::max(0.001, move_step * 0.5);
        if (key == ']') move_step = std::min(0.100, move_step * 2.0);
        if (key == ',') size_step = std::max(0.001, size_step * 0.5);
        if (key == '.') size_step = std::min(0.100, size_step * 2.0);

        if (selected_roi == '1') nudge_ratio(rois.red_roi, key, move_step, size_step);
        else if (selected_roi == '2') nudge_ratio(rois.image_roi, key, move_step, size_step);

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
    if (lock.valid) save_homography_json(dep_cfg.save_h_path, lock);
    if (dep_cfg.save_remap_cache && remap_cache.valid) save_warp_remap_cache(dep_cfg.save_remap_path, remap_cache);
    save_rois_json(dep_cfg.save_rois_path, rois);
    return true;
}

} // namespace vision_app
