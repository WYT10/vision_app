#pragma once

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "app_config.hpp"
#include "app_types.hpp"
#include "camera.hpp"
#include "calibrate.hpp"
#include "model.hpp"
#include "stats.hpp"
#include "status_ui.hpp"
#include "trigger.hpp"

namespace vision_app {
namespace {

inline bool wants_overlay(const AppOptions& opt) {
    return opt.draw_overlay && (opt.text_sink == "overlay" || opt.text_sink == "split");
}

inline bool wants_status_window(const AppOptions& opt) {
    return opt.ui && opt.show_status_window && (opt.text_sink == "status_window" || opt.text_sink == "split");
}

inline bool wants_terminal(const AppOptions& opt) {
    return opt.text_sink == "terminal" || opt.text_sink == "split";
}

inline std::string onoff(bool v) { return v ? "on" : "off"; }

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

inline std::vector<std::string> build_calibrate_status_lines(const AppOptions& opt,
                                                             bool locked,
                                                             int selected,
                                                             double move_step,
                                                             double size_step,
                                                             const TriggerDebugInfo& dbg,
                                                             const ModelResult& model_res,
                                                             const RoiConfig& rois,
                                                             const DynamicRedStackedConfig& dyn_cfg) {
    std::vector<std::string> lines;
    lines.push_back(std::string("mode: calibrate  trigger=") + opt.trigger_mode + "  lock=" + (locked ? "locked" : "search"));
    lines.push_back("hotkeys: enter/space lock, u unlock, p save all, y save warp, o save profile, q quit");
    if (opt.trigger_mode == "fixed_rect") {
        lines.push_back("edit targets: 1 red_roi, 2 image_roi");
        std::ostringstream a, b;
        a << "red_roi=" << rois.red_roi.x << "," << rois.red_roi.y << "," << rois.red_roi.w << "," << rois.red_roi.h;
        b << "image_roi=" << rois.image_roi.x << "," << rois.image_roi.y << "," << rois.image_roi.w << "," << rois.image_roi.h;
        lines.push_back(a.str());
        lines.push_back(b.str());
        lines.push_back(std::string("selected=") + (selected == 0 ? "red_roi" : "image_roi"));
        std::ostringstream c;
        c << "red_ratio=" << dbg.red_ratio << "  threshold=" << opt.fixed_cfg.red_ratio_threshold << "  triggered=" << (dbg.triggered ? 1 : 0);
        lines.push_back(c.str());
    } else {
        lines.push_back("edit targets: 1 upper_band, 2 lower_band, 3 derived image_roi");
        std::ostringstream a, b, c;
        a << "upper_band=" << dyn_cfg.upper_band.y << "," << dyn_cfg.upper_band.h;
        b << "lower_band=" << dyn_cfg.lower_band.y << "," << dyn_cfg.lower_band.h;
        c << "image(bottom_offset,width,height)=" << dyn_cfg.image_roi.bottom_offset << "," << dyn_cfg.image_roi.width << "," << dyn_cfg.image_roi.height;
        lines.push_back(a.str());
        lines.push_back(b.str());
        lines.push_back(c.str());
        lines.push_back(std::string("selected=") + (selected == 0 ? "upper_band" : selected == 1 ? "lower_band" : "image_roi"));
        std::ostringstream d;
        d << "upper(fill=" << dbg.upper_fill_ratio << ", w=" << dbg.upper_width_ratio << ")  lower(fill=" << dbg.lower_fill_ratio << ", w=" << dbg.lower_width_ratio << ")";
        lines.push_back(d.str());
        std::ostringstream e;
        e << "cx=" << dbg.center_x_px << "  triggered=" << (dbg.triggered ? 1 : 0);
        lines.push_back(e.str());
    }
    std::ostringstream st;
    st << "move_step=" << move_step << " size_step=" << size_step << " model=" << (model_res.ran ? model_res.summary : "idle");
    lines.push_back(st.str());
    return lines;
}

inline std::vector<std::string> build_deploy_status_lines(const AppOptions& opt,
                                                          const TriggerDebugInfo& dbg,
                                                          const ModelResult& model_res,
                                                          double fps,
                                                          double warp_ms,
                                                          double roi_ms) {
    std::vector<std::string> lines;
    lines.push_back(std::string("mode: deploy  trigger=") + opt.trigger_mode + "  backend=" + opt.model_cfg.backend);
    lines.push_back(std::string("red=") + onoff(opt.run_red) + " image_roi=" + onoff(opt.run_image_roi) + " model=" + onoff(opt.run_model));
    if (opt.trigger_mode == "fixed_rect") {
        std::ostringstream a;
        a << "red_ratio=" << dbg.red_ratio << "  threshold=" << opt.fixed_cfg.red_ratio_threshold << "  triggered=" << (dbg.triggered ? 1 : 0);
        lines.push_back(a.str());
    } else {
        std::ostringstream a, b;
        a << "upper(fill=" << dbg.upper_fill_ratio << ", w=" << dbg.upper_width_ratio << ")  lower(fill=" << dbg.lower_fill_ratio << ", w=" << dbg.lower_width_ratio << ")";
        b << "cx=" << dbg.center_x_px << "  triggered=" << (dbg.triggered ? 1 : 0);
        lines.push_back(a.str());
        lines.push_back(b.str());
    }
    lines.push_back(std::string("model: ") + (model_res.ran ? model_res.summary : "idle"));
    std::ostringstream perf;
    perf << "fps=" << fps << " warp_ms=" << warp_ms << " roi_ms=" << roi_ms << " infer_ms=" << model_res.infer_ms;
    lines.push_back(perf.str());
    lines.push_back("quit: q / ESC");
    return lines;
}

inline bool save_all_outputs(const AppOptions& opt, const RoiConfig& rois, const WarpPackage& pack, std::string& log_msg) {
    std::string err;
    const bool ok_profile = save_profile_config(opt.profile_path, opt, err);
    const bool ok_rois = save_rois_yaml(opt.save_rois, rois);
    const bool ok_warp = save_warp_package(opt.save_warp, pack);
    const bool ok_report = write_report_md(opt.save_report, "Calibration Report", nullptr,
                                           std::string("- trigger_mode: ") + opt.trigger_mode + "\n" +
                                           "- warp: " + opt.save_warp + "\n" +
                                           "- profile: " + opt.profile_path + "\n");
    std::ostringstream oss;
    oss << "save profile=" << ok_profile << " rois=" << ok_rois << " warp=" << ok_warp << " report=" << ok_report;
    if (!err.empty()) oss << " err=" << err;
    log_msg = oss.str();
    return ok_profile && ok_rois && ok_warp;
}

} // namespace

inline bool run_probe(const AppOptions& opt, std::string& err) {
    CameraProbeResult probe;
    if (!probe_camera(opt.device, probe, err)) return false;
    print_probe(probe);
    return true;
}

inline bool run_live(const AppOptions& opt, std::string& err) {
    RuntimeStats stats;
    if (!bench_capture(opt.device, opt.width, opt.height, opt.fps, opt.fourcc, opt.buffer_size,
                       opt.latest_only, opt.drain_grabs, !opt.ui, opt.duration,
                       opt.camera_soft_max, opt.camera_preview_max, stats, err)) return false;
    print_runtime_stats(stats);
    return true;
}

inline bool run_calibrate(AppOptions& opt, std::string& err) {
    int cam_w = opt.width, cam_h = opt.height;
    cv::VideoCapture cap;
    if (!open_capture(cap, opt.device, cam_w, cam_h, opt.fps, opt.fourcc, opt.buffer_size, opt.camera_soft_max, err)) return false;

    RoiConfig rois = opt.default_rois;
    (void)load_rois_yaml(opt.load_rois, rois);

    bool model_ready = false;
    if (opt.model_cfg.enable && opt.run_model) {
        std::string merr;
        model_ready = init_model_runtime(opt.model_cfg, merr);
        if (!model_ready && wants_terminal(opt)) std::cerr << "[model] disabled: " << merr << "\n";
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
    AprilTagDetection cur, locked_det;
    WarpPackage locked_pack;
    ModelResult model_res;
    RoiRuntimeData roi_info;
    TriggerDebugInfo dbg;
    cv::Mat frame, camera_show, warp_show, temp_preview, warped, valid;

    if (opt.ui) {
        cv::namedWindow("vision_app_camera", cv::WINDOW_NORMAL);
        cv::namedWindow("vision_app_warp", cv::WINDOW_NORMAL);
        cv::resizeWindow("vision_app_camera", opt.camera_preview_max, std::max(240, opt.camera_preview_max * 3 / 4));
        cv::resizeWindow("vision_app_warp", opt.warp_preview_max, opt.warp_preview_max);
        if (wants_status_window(opt)) cv::namedWindow("vision_app_status", cv::WINDOW_NORMAL);
        if (wants_terminal(opt)) {
            std::cout << "=== calibrate controls ===\n"
                      << "enter/space lock | u unlock | p save all | y save warp | o save profile | q quit\n"
                      << "fixed_rect: 1 red_roi, 2 image_roi\n"
                      << "dynamic_red_stacked: 1 upper_band, 2 lower_band, 3 derived image_roi\n"
                      << "move: wasd | size: ijkl | steps: [ ] and , .\n";
        }
    }

    while (true) {
        if (!grab_latest_frame(cap, opt.latest_only, opt.drain_grabs, frame)) { err = "failed to read frame"; break; }

        if (!locked) {
            std::string derr;
            detect_apriltag_best(frame, tag_cfg, cur, derr);
            const bool stable = locker.update(cur);
            if (!opt.manual_lock_only && stable && cur.found) {
                if (build_warp_package_from_detection(cur, frame.size(), opt.warp_width, opt.warp_height, opt.target_tag_px, locked_pack, err)) {
                    locked = true;
                    locked_det = cur;
                    if (wants_terminal(opt)) std::cout << "[lock] auto family=" << locked_det.family << " id=" << locked_det.id << "\n";
                }
            }

            camera_show = frame.clone();
            if (wants_overlay(opt)) {
                draw_detection_overlay(camera_show, cur);
                cv::putText(camera_show, cur.found ? "SEARCH" : "NO TAG", {12, 56}, cv::FONT_HERSHEY_SIMPLEX, 0.62,
                            cur.found ? cv::Scalar(0,255,255) : cv::Scalar(0,0,255), 2);
            }

            temp_preview.release();
            if (cur.found) {
                std::string werr;
                WarpPackage temp_pack;
                if (build_warp_package_from_detection(cur, frame.size(), opt.warp_width, opt.warp_height, opt.target_tag_px, temp_pack, werr)) {
                    cv::Mat temp_valid;
                    if (apply_warp(frame, temp_pack, temp_preview, &temp_valid)) {
                        if (opt.trigger_mode == "fixed_rect") {
                            std::string rerr;
                            extract_runtime_rois_fixed(temp_preview, temp_valid, rois, opt.red_cfg, opt.fixed_cfg, roi_info, dbg, rerr);
                            if (wants_overlay(opt)) draw_trigger_overlay_fixed(temp_preview, rois, dbg, selected == 0, selected == 1);
                        } else {
                            std::string rerr;
                            extract_runtime_rois_dynamic(temp_preview, temp_valid, opt.dynamic_cfg, opt.red_cfg, roi_info, dbg, rerr);
                            if (wants_overlay(opt)) draw_trigger_overlay_dynamic(temp_preview, dbg, selected == 0, selected == 1, selected == 2);
                        }
                    }
                }
            }
            warp_show = temp_preview.empty() ? make_blank_preview(opt.warp_width, opt.warp_height, "waiting for tag") : temp_preview.clone();
        } else {
            cur = locked_det;
            if (!apply_warp(frame, locked_pack, warped, &valid)) { err = "failed to apply locked warp"; break; }
            camera_show = frame.clone();
            warp_show = warped.clone();

            if (wants_overlay(opt)) {
                draw_detection_overlay(camera_show, locked_det);
                cv::putText(camera_show, "LOCKED", {12, 56}, cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(0,255,0), 2);
            }

            std::string rerr;
            const bool ok = (opt.trigger_mode == "fixed_rect")
                ? extract_runtime_rois_fixed(warped, valid, rois, opt.red_cfg, opt.fixed_cfg, roi_info, dbg, rerr)
                : extract_runtime_rois_dynamic(warped, valid, opt.dynamic_cfg, opt.red_cfg, roi_info, dbg, rerr);
            if (ok && model_ready && opt.run_model && opt.run_image_roi) {
                std::string merr;
                if (!run_model_on_image_roi(roi_info, opt.model_cfg, model_res, merr)) {
                    model_res = {};
                    model_res.ran = true;
                    model_res.summary = "model err: " + merr;
                }
            } else if (!ok) {
                model_res = {};
                model_res.summary = rerr;
            }

            if (wants_overlay(opt)) {
                if (opt.trigger_mode == "fixed_rect") draw_trigger_overlay_fixed(warp_show, rois, dbg, selected == 0, selected == 1);
                else draw_trigger_overlay_dynamic(warp_show, dbg, selected == 0, selected == 1, selected == 2);
                if (model_res.ran) cv::putText(warp_show, model_res.summary, {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(0,120,0), 1);
            }
        }

        if (opt.ui) {
            cv::imshow("vision_app_camera", downscale_for_preview(camera_show, opt.camera_preview_max));
            cv::imshow("vision_app_warp", downscale_for_preview(warp_show, opt.warp_preview_max));
            if (wants_status_window(opt)) {
                const auto lines = build_calibrate_status_lines(opt, locked, selected, move_step, size_step, dbg, model_res, rois, opt.dynamic_cfg);
                cv::imshow("vision_app_status", build_status_panel("vision_app status", lines, opt.status_width));
            }
        }

        ++frame_idx;
        int key = opt.ui ? (cv::waitKey(1) & 0xFF) : -1;
        if (key == 27 || key == 'q') break;
        if (key == 'u') {
            locked = false;
            locker.reset();
            locked_pack = {};
            model_res = {};
            dbg = {};
            if (wants_terminal(opt)) std::cout << "[lock] released\n";
            continue;
        }
        if ((key == ' ' || key == 13) && cur.found) {
            if (build_warp_package_from_detection(cur, frame.size(), opt.warp_width, opt.warp_height, opt.target_tag_px, locked_pack, err)) {
                locked = true;
                locked_det = cur;
                if (wants_terminal(opt)) std::cout << "[lock] manual family=" << locked_det.family << " id=" << locked_det.id << "\n";
            } else if (wants_terminal(opt)) {
                std::cerr << "[lock] rejected: " << err << "\n";
                err.clear();
            }
        }

        if (key == '[') move_step = std::max(0.001, move_step * 0.5);
        if (key == ']') move_step = std::min(0.25, move_step * 2.0);
        if (key == ',') size_step = std::max(0.001, size_step * 0.5);
        if (key == '.') size_step = std::min(0.25, size_step * 2.0);
        if (key == 'r') {
            rois = opt.default_rois;
            opt.dynamic_cfg = DynamicRedStackedConfig{};
            if (wants_terminal(opt)) std::cout << "[reset] trigger geometry reset\n";
        }

        if (opt.trigger_mode == "fixed_rect") {
            if (key == '1') selected = 0;
            if (key == '2') selected = 1;
            if (locked) {
                if (selected == 0) adjust_roi(rois.red_roi, key, move_step, size_step);
                else adjust_roi(rois.image_roi, key, move_step, size_step);
            }
        } else {
            if (key == '1') selected = 0;
            if (key == '2') selected = 1;
            if (key == '3') selected = 2;
            if (locked) {
                if (selected == 0) adjust_band(opt.dynamic_cfg.upper_band, key, move_step, size_step);
                else if (selected == 1) adjust_band(opt.dynamic_cfg.lower_band, key, move_step, size_step);
                else adjust_dynamic_image_roi(opt.dynamic_cfg.image_roi, key, move_step, size_step);
            }
        }

        if (key == 'o') {
            opt.default_rois = rois;
            std::string serr;
            const bool ok = save_profile_config(opt.profile_path, opt, serr);
            if (wants_terminal(opt)) std::cout << "[save] profile=" << ok << (serr.empty() ? "" : (" err=" + serr)) << "\n";
        }
        if (key == 'y' && locked) {
            const bool ok = save_warp_package(opt.save_warp, locked_pack);
            if (wants_terminal(opt)) std::cout << "[save] warp=" << ok << " path=" << opt.save_warp << "\n";
        }
        if (key == 'p' && locked) {
            opt.default_rois = rois;
            std::string msg;
            save_all_outputs(opt, rois, locked_pack, msg);
            if (wants_terminal(opt)) std::cout << "[save] " << msg << "\n";
        }
    }

    release_model_runtime();
    if (opt.ui) cv::destroyAllWindows();
    return err.empty();
}

inline bool run_deploy(AppOptions& opt, std::string& err) {
    WarpPackage pack;
    if (!load_warp_package(opt.load_warp, pack)) {
        err = "failed to load warp package: " + opt.load_warp;
        return false;
    }

    RoiConfig rois = opt.default_rois;
    (void)load_rois_yaml(opt.load_rois, rois);

    bool model_ready = false;
    if (opt.model_cfg.enable && opt.run_model) {
        std::string merr;
        model_ready = init_model_runtime(opt.model_cfg, merr);
        if (!model_ready && wants_terminal(opt)) std::cerr << "[model] disabled: " << merr << "\n";
    }

    int cam_w = opt.width, cam_h = opt.height;
    cv::VideoCapture cap;
    if (!open_capture(cap, opt.device, cam_w, cam_h, opt.fps, opt.fourcc, opt.buffer_size, opt.camera_soft_max, err)) return false;

    cv::Mat frame, warped, valid, camera_show, warp_show;
    RoiRuntimeData roi_info;
    ModelResult model_res;
    TriggerDebugInfo dbg;

    if (opt.ui) {
        cv::namedWindow("vision_app_camera", cv::WINDOW_NORMAL);
        cv::namedWindow("vision_app_warp", cv::WINDOW_NORMAL);
        cv::resizeWindow("vision_app_camera", opt.camera_preview_max, std::max(240, opt.camera_preview_max * 3 / 4));
        cv::resizeWindow("vision_app_warp", opt.warp_preview_max, opt.warp_preview_max);
        if (wants_status_window(opt)) cv::namedWindow("vision_app_status", cv::WINDOW_NORMAL);
    }

    using Clock = std::chrono::steady_clock;
    auto loop_t0 = Clock::now();
    Clock::time_point last_model_tp{};
    bool have_last_model_tp = false;
    int frame_idx = 0;

    while (true) {
        auto frame_t0 = Clock::now();
        if (!grab_latest_frame(cap, opt.latest_only, opt.drain_grabs, frame)) { err = "failed to read frame"; break; }

        auto warp_t0 = Clock::now();
        if (!apply_warp(frame, pack, warped, &valid)) { err = "failed to apply warp"; break; }
        const double warp_ms = std::chrono::duration<double, std::milli>(Clock::now() - warp_t0).count();

        camera_show = frame.clone();
        warp_show = warped.clone();

        std::string rerr;
        auto roi_t0 = Clock::now();
        const bool roi_ok = (opt.trigger_mode == "fixed_rect")
            ? extract_runtime_rois_fixed(warped, valid, rois, opt.red_cfg, opt.fixed_cfg, roi_info, dbg, rerr)
            : extract_runtime_rois_dynamic(warped, valid, opt.dynamic_cfg, opt.red_cfg, roi_info, dbg, rerr);
        const double roi_ms = std::chrono::duration<double, std::milli>(Clock::now() - roi_t0).count();

        const bool stride_ok = (opt.model_cfg.stride <= 1) || ((frame_idx % opt.model_cfg.stride) == 0);
        bool hz_ok = true;
        if (opt.model_max_hz > 0.0 && have_last_model_tp) {
            const double dt = std::chrono::duration<double>(Clock::now() - last_model_tp).count();
            hz_ok = dt >= (1.0 / opt.model_max_hz);
        }

        if (roi_ok && model_ready && opt.run_model && opt.run_image_roi && stride_ok && hz_ok) {
            std::string merr;
            if (!run_model_on_image_roi(roi_info, opt.model_cfg, model_res, merr)) {
                model_res = {};
                model_res.ran = true;
                model_res.summary = "model err: " + merr;
            }
            last_model_tp = Clock::now();
            have_last_model_tp = true;
        }

        if (roi_ok && opt.save_every_n > 0 && (frame_idx % opt.save_every_n) == 0) {
            if (opt.run_image_roi) save_crop_if_needed(opt.save_image_roi_dir, "image_roi", frame_idx, roi_info.image_bgr);
            if (opt.run_red) save_crop_if_needed(opt.save_red_roi_dir, "red_roi", frame_idx, roi_info.red_bgr);
        }

        if (wants_overlay(opt)) {
            cv::putText(camera_show, "DEPLOY", {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
            if (opt.trigger_mode == "fixed_rect") draw_trigger_overlay_fixed(warp_show, rois, dbg, false, false);
            else draw_trigger_overlay_dynamic(warp_show, dbg, false, false, false);
            if (model_res.ran) cv::putText(warp_show, model_res.summary, {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(0,120,0), 1);
        }

        const double total_ms = std::chrono::duration<double, std::milli>(Clock::now() - frame_t0).count();
        const double fps = (total_ms > 0.0) ? (1000.0 / total_ms) : 0.0;

        if (opt.ui) {
            cv::imshow("vision_app_camera", downscale_for_preview(camera_show, opt.camera_preview_max));
            cv::imshow("vision_app_warp", downscale_for_preview(warp_show, opt.warp_preview_max));
            if (wants_status_window(opt)) {
                const auto lines = build_deploy_status_lines(opt, dbg, model_res, fps, warp_ms, roi_ms);
                cv::imshow("vision_app_status", build_status_panel("vision_app status", lines, opt.status_width));
            }
        }

        ++frame_idx;
        const int key = opt.ui ? (cv::waitKey(1) & 0xFF) : -1;
        if (key == 27 || key == 'q') break;
        if (!opt.ui && opt.duration > 0) {
            const double elapsed = std::chrono::duration<double>(Clock::now() - loop_t0).count();
            if (elapsed >= opt.duration) break;
        }
    }

    release_model_runtime();
    if (opt.ui) cv::destroyAllWindows();
    return err.empty();
}

} // namespace vision_app
