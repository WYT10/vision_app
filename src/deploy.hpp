#pragma once

#include <chrono>
#include <filesystem>
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

    int warp_width = 720;      // tag rect target width before full-view bbox translation
    int warp_height = 720;     // tag rect target height before full-view bbox translation
    int warp_view_max_side = 900;

    bool live_preview_raw = true;
    bool live_preview_warp = true;
    bool show_help_overlay = true;
    bool show_status_overlay = true;
    bool save_snapshots = false;
    std::string snapshot_dir = "../report";

    bool use_remap_cache = true;
    bool fixed_point_remap = true;
    bool save_remap_cache = true;

    int interactive_max_side = 1000;
    bool unsafe_big_frame = false;

    double move_step = 0.005;
    double size_step = 0.005;

    double red_mean_threshold = 120.0;
    double red_dominance_threshold = 20.0;
    int trigger_cooldown_frames = 10;
    bool save_triggered_image_roi = true;
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
        << "\nLive/deploy keys:\n"
        << "  q / ESC      quit\n"
        << "  h            toggle help overlay\n"
        << "  space/enter  lock current visible tag\n"
        << "  u            unlock and reacquire\n"
        << "  p            save all outputs\n"
        << "  y            save homography only\n"
        << "  o            save rois only\n"
        << "  g            save remap cache only\n"
        << "  1 / 2        select red_roi / image_roi\n"
        << "  wasd         move selected roi\n"
        << "  ijkl         resize selected roi\n"
        << "  [ / ]        adjust move step\n"
        << "  , / .        adjust size step\n"
        << "  r            reset selected roi to defaults\n"
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
                          char selected_roi,
                          bool show_help_overlay,
                          bool auto_save_lock,
                          bool use_remap_cache,
                          double move_step,
                          double size_step,
                          const RedGateResult* gate,
                          const InferStubResult* infer,
                          const DeployConfig& dep_cfg) {
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
                std::string("  max_view_side=") + std::to_string(dep_cfg.warp_view_max_side),
            std::string("red gate: mean_r>=") + std::to_string(static_cast<int>(dep_cfg.red_mean_threshold)) +
                " and red_score>=" + std::to_string(static_cast<int>(dep_cfg.red_dominance_threshold))
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
        if (gate && gate->valid) {
            std::ostringstream gs;
            gs << "gate=" << (gate->triggered ? "TRIGGER" : "idle")
               << "  mean_r=" << std::fixed << std::setprecision(1) << gate->mean_r
               << "  red_score=" << gate->red_score;
            cv::putText(*warped_view, gs.str(), {20, warped_view->rows - 46}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        gate->triggered ? cv::Scalar(0,255,255) : cv::Scalar(200,200,200), 1);
        }
        if (infer && infer->valid) {
            cv::putText(*warped_view, infer->summary, {20, warped_view->rows - 68}, cv::FONT_HERSHEY_SIMPLEX, 0.46, {230,230,230}, 1);
        }
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
    if (dep_cfg.save_remap_cache && remap_cache && remap_cache->valid) ok = save_warp_remap_cache(dep_cfg.save_remap_path, *remap_cache) && ok;
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
    if (dep_cfg.save_remap_cache) save_warp_remap_cache(dep_cfg.save_remap_path, cache);
    return true;
}

static bool validate_interactive_camera_size(const CameraConfig& cam_cfg,
                                             const DeployConfig& dep_cfg,
                                             std::string& err) {
    if (cam_cfg.headless) return true;
    if (dep_cfg.unsafe_big_frame) return true;
    const int long_side = std::max(cam_cfg.width, cam_cfg.height);
    if (long_side > dep_cfg.interactive_max_side) {
        std::ostringstream oss;
        oss << "interactive mode blocked because requested camera size " << cam_cfg.width << "x" << cam_cfg.height
            << " exceeds safe max side " << dep_cfg.interactive_max_side
            << ". Use bench for larger modes or pass --unsafe-big-frame 1.";
        err = oss.str();
        return false;
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

static std::string make_trigger_path(const DeployConfig& dep_cfg, uint64_t trigger_index) {
    std::ostringstream oss;
    oss << dep_cfg.snapshot_dir << "/trigger_image_roi_" << std::setw(6) << std::setfill('0') << trigger_index << ".png";
    return oss.str();
}

static bool run_live_or_deploy_mode(const CameraConfig& cam_cfg,
                                    const AprilTagConfig& tag_cfg,
                                    const DeployConfig& dep_cfg,
                                    RoiSet& rois,
                                    CaptureStats& out_capture,
                                    StageStats& out_stage,
                                    HomographyLock& out_lock,
                                    std::string& err) {
    if (!validate_interactive_camera_size(cam_cfg, dep_cfg, err)) return false;

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
    RollingTimer detect_t, warp_t, infer_t, total_t;
    cv::Mat frame, raw_view, warped;
    WarpRemapCache remap_cache;
    char selected_roi = '1';
    bool show_help_overlay = dep_cfg.show_help_overlay;
    bool auto_save_lock = dep_cfg.auto_save_lock;
    double move_step = dep_cfg.move_step;
    double size_step = dep_cfg.size_step;
    int cooldown = 0;
    uint64_t trigger_count = 0;
    RedGateResult last_gate{};
    InferStubResult last_infer{};
    print_live_keys();

    if (!cam_cfg.headless) {
        if (dep_cfg.live_preview_raw) cv::namedWindow("vision_app_raw", cv::WINDOW_AUTOSIZE);
        if (dep_cfg.live_preview_warp) cv::namedWindow("vision_app_warp", cv::WINDOW_AUTOSIZE);
    }

    const auto t0 = std::chrono::steady_clock::now();
    while (true) {
        const auto loop_start = std::chrono::steady_clock::now();
        if (!cam.read_latest(frame, err)) return false;
        if (lock.valid && lock.source_size.width > 0 && lock.source_size.height > 0 && frame.size() != lock.source_size) {
            std::ostringstream oss;
            oss << "saved calibration expects source size " << lock.source_size.width << "x" << lock.source_size.height
                << " but current camera delivers " << frame.cols << "x" << frame.rows;
            err = oss.str();
            return false;
        }

        raw_view = frame.clone();
        AprilTagDetection det;

        if (!lock.valid) {
            const auto td0 = std::chrono::steady_clock::now();
            if (!detect_apriltag_best(frame, tag_cfg, det, err)) return false;
            detect_t.push(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count());

            const bool allow_auto_lock = !tag_cfg.manual_lock_only;
            const bool got_lock = det.found && locker.update(det, frame.size(), allow_auto_lock);
            draw_detection_overlay(raw_view, det, got_lock);
            if (got_lock) {
                if (!compute_full_view_homography_from_tag_quad(locker.locked_det(), frame.size(), cv::Size(dep_cfg.warp_width, dep_cfg.warp_height), dep_cfg.warp_view_max_side, lock)) {
                    err = "failed to compute full-view homography after tag lock";
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
            const bool warped_ok = dep_cfg.use_remap_cache ? warp_full_frame_cached(frame, warped, remap_cache) : warp_full_frame(frame, warped, lock);
            if (!warped_ok) {
                err = dep_cfg.use_remap_cache ? "remap warp failed" : "warpPerspective failed";
                return false;
            }
            warp_t.push(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw0).count());

            last_gate = evaluate_red_gate(crop_roi_clone(warped, rois.red_roi), dep_cfg.red_mean_threshold, dep_cfg.red_dominance_threshold);
            if (cooldown > 0) --cooldown;
            if (last_gate.valid && last_gate.triggered && cooldown <= 0) {
                const auto ti0 = std::chrono::steady_clock::now();
                const cv::Mat image_roi = crop_roi_clone(warped, rois.image_roi);
                last_infer = run_infer_stub(image_roi);
                infer_t.push(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ti0).count());
                ++trigger_count;
                cooldown = std::max(0, dep_cfg.trigger_cooldown_frames);
                if (dep_cfg.save_triggered_image_roi && !image_roi.empty()) {
                    std::filesystem::create_directories(dep_cfg.snapshot_dir);
                    cv::imwrite(make_trigger_path(dep_cfg, trigger_count), image_roi);
                }
            }

            draw_roi_overlay(warped, rois, lock, selected_roi, &last_gate, &last_infer);
            draw_live_hud(raw_view, &warped, tag_cfg, locker, det, lock, selected_roi, show_help_overlay, auto_save_lock, dep_cfg.use_remap_cache, move_step, size_step, &last_gate, &last_infer, dep_cfg);
        } else {
            draw_live_hud(raw_view, nullptr, tag_cfg, locker, det, lock, selected_roi, show_help_overlay, auto_save_lock, dep_cfg.use_remap_cache, move_step, size_step, nullptr, nullptr, dep_cfg);
        }

        if (!cam_cfg.headless && dep_cfg.live_preview_raw) cv::imshow("vision_app_raw", raw_view);
        if (!cam_cfg.headless && dep_cfg.live_preview_warp && !warped.empty()) cv::imshow("vision_app_warp", warped);

        total_t.push(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - loop_start).count());

        const int key = (cam_cfg.headless ? -1 : (cv::waitKey(1) & 0xFF));
        if (key == 'q' || key == 27) break;
        if (key == '1') selected_roi = '1';
        if (key == '2') selected_roi = '2';
        if (key == 'h') show_help_overlay = !show_help_overlay;
        if (key == 'u') { lock = {}; locker.reset(); warped.release(); remap_cache = {}; last_gate = {}; last_infer = {}; }
        if (key == 'r') {
            if (selected_roi == '1') rois.red_roi = default_red_roi();
            else rois.image_roi = default_image_roi();
        }
        if (key == 'p') {
            StageStats stage_preview{};
            stage_preview.frames = cam.stats().frames;
            stage_preview.detect_ms_avg = detect_t.avg();
            stage_preview.warp_ms_avg = warp_t.avg();
            stage_preview.infer_ms_avg = infer_t.avg();
            stage_preview.total_ms_avg = total_t.avg();
            stage_preview.detect_ms_max = detect_t.max_ms;
            stage_preview.warp_ms_max = warp_t.max_ms;
            stage_preview.infer_ms_max = infer_t.max_ms;
            stage_preview.total_ms_max = total_t.max_ms;
            stage_preview.detector_fps_avg = (stage_preview.detect_ms_avg > 0.0) ? (1000.0 / stage_preview.detect_ms_avg) : 0.0;
            stage_preview.trigger_count = trigger_count;
            stage_preview.last_red_mean = last_gate.mean_r;
            stage_preview.last_red_score = last_gate.red_score;
            save_all_outputs(dep_cfg, cam_cfg, cam.stats(), stage_preview, lock.valid ? &lock : nullptr, rois, remap_cache.valid ? &remap_cache : nullptr);
            std::cout << "Saved all outputs\n";
        }
        if (key == 'y' && lock.valid) { save_homography_json(dep_cfg.save_h_path, lock); std::cout << "Saved homography\n"; }
        if (key == 'o') { save_rois_json(dep_cfg.save_rois_path, rois); std::cout << "Saved rois\n"; }
        if (key == 'g' && remap_cache.valid) { save_warp_remap_cache(dep_cfg.save_remap_path, remap_cache); std::cout << "Saved remap cache\n"; }
        if ((key == ' ' || key == 13) && !lock.valid) {
            const AprilTagDetection cand = locker.last_candidate().found ? locker.last_candidate() : det;
            if (locker.force_lock(cand, frame.size())) {
                if (!compute_full_view_homography_from_tag_quad(locker.locked_det(), frame.size(), cv::Size(dep_cfg.warp_width, dep_cfg.warp_height), dep_cfg.warp_view_max_side, lock)) {
                    err = "failed to compute full-view homography after manual lock";
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
            std::filesystem::create_directories(dep_cfg.snapshot_dir);
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
    out_stage.infer_ms_avg = infer_t.avg();
    out_stage.total_ms_avg = total_t.avg();
    out_stage.detect_ms_max = detect_t.max_ms;
    out_stage.warp_ms_max = warp_t.max_ms;
    out_stage.infer_ms_max = infer_t.max_ms;
    out_stage.total_ms_max = total_t.max_ms;
    out_stage.detector_fps_avg = (out_stage.detect_ms_avg > 0.0) ? (1000.0 / out_stage.detect_ms_avg) : 0.0;
    out_stage.trigger_count = trigger_count;
    out_stage.last_red_mean = last_gate.mean_r;
    out_stage.last_red_score = last_gate.red_score;
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
