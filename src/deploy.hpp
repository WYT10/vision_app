#pragma once

#include <filesystem>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "camera.hpp"
#include "calibrate.hpp"
#include "stats.hpp"

namespace vision_app {

struct AppOptions {
    std::string mode = "probe";
    std::string device = "/dev/video0";
    int width = 640;
    int height = 480;
    int fps = 120;
    std::string fourcc = "MJPG";
    int buffer_size = 1;
    bool latest_only = true;
    int drain_grabs = 1;
    bool headless = false;
    int duration = 10;

    int camera_soft_max = 1000;
    int warp_soft_max = 700;
    int preview_soft_max = 500;
    int temp_preview_square = 220;
    int temp_preview_stride = 3;

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
};

inline void draw_help_overlay(cv::Mat& img, bool locked, int selected, double move_step, double size_step) {
    const int base_x = 12;
    int y = img.rows - 120;
    cv::putText(img, locked ? "LOCKED / ROI EDIT" : "SEARCHING / LIVE PREVIEW", {base_x, y}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(30,255,30), 2); y += 22;
    cv::putText(img, "SPACE lock   U unlock   H help   Q quit", {base_x, y}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1); y += 18;
    cv::putText(img, "1 red_roi   2 image_roi   selected=" + std::string(selected==0?"red":"image"), {base_x, y}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1); y += 18;
    cv::putText(img, "WASD move   I/K height   J/L width", {base_x, y}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1); y += 18;
    cv::putText(img, "[ ] move step=" + std::to_string(move_step).substr(0,5) + "   , . size step=" + std::to_string(size_step).substr(0,5), {base_x, y}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1); y += 18;
    cv::putText(img, "P save all   O save rois   Y save warp   R reset", {base_x, y}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);
}

inline cv::Mat make_temp_live_preview(const cv::Mat& frame,
                                      const AprilTagDetection& det,
                                      int temp_square,
                                      int warp_soft_max) {
    if (!det.found) return cv::Mat();
    WarpPackage temp_pack;
    std::string err;
    if (!build_warp_package_from_detection(det, frame.size(), std::min(temp_square, warp_soft_max), temp_pack, err)) return cv::Mat();
    cv::Mat warped, valid;
    if (!apply_warp(frame, temp_pack, warped, &valid)) return cv::Mat();
    cv::Mat show;
    if (warped.channels()==1) cv::cvtColor(warped, show, cv::COLOR_GRAY2BGR);
    else show = warped;
    return show;
}

inline bool run_live(const AppOptions& opt, std::string& err) {
    int cam_w = opt.width, cam_h = opt.height;
    cv::VideoCapture cap;
    if (!open_capture(cap, opt.device, cam_w, cam_h, opt.fps, opt.fourcc, opt.buffer_size, opt.camera_soft_max, err)) return false;

    RoiConfig rois = opt.default_rois;
    (void)load_rois_yaml(opt.load_rois, rois); // optional

    AprilTagConfig tag_cfg; tag_cfg.family = opt.tag_family; tag_cfg.target_id = opt.target_id; tag_cfg.require_target_id = opt.require_target_id; tag_cfg.manual_lock_only = opt.manual_lock_only; tag_cfg.lock_frames = opt.lock_frames;
    TagLocker locker(opt.lock_frames);

    bool locked = false;
    bool show_help = true;
    int selected = 0;
    double move_step = 0.01;
    double size_step = 0.01;
    int frame_idx = 0;
    AprilTagDetection cur, locked_det;
    WarpPackage locked_pack;
    cv::Mat frame, display, temp_preview, warped, valid;

    cv::namedWindow("vision_app", cv::WINDOW_AUTOSIZE);

    while (true) {
        if (!grab_latest_frame(cap, opt.latest_only, opt.drain_grabs, frame)) { err = "failed to read frame"; return false; }

        if (!locked) {
            std::string derr;
            detect_apriltag_best(frame, tag_cfg, cur, derr);
            const bool stable = locker.update(cur);
            if (!opt.manual_lock_only && stable && cur.found) {
                if (build_warp_package_from_detection(cur, frame.size(), opt.warp_soft_max, locked_pack, err)) {
                    locked = true;
                    locked_det = cur;
                }
                err.clear();
            }

            cv::Mat raw = frame.clone();
            draw_detection_overlay(raw, cur);
            std::string status = cur.found ? ("SEARCH family=" + cur.family + " id=" + std::to_string(cur.id) + " [SPACE lock]") : "SEARCH no tag";
            cv::putText(raw, status, {12, 56}, cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(0,255,255), 2);

            if (cur.found && (frame_idx % std::max(1,opt.temp_preview_stride) == 0)) {
                temp_preview = make_temp_live_preview(frame, cur, opt.temp_preview_square, opt.warp_soft_max);
            }
            if (!temp_preview.empty()) {
                cv::Mat inset = downscale_for_preview(temp_preview, opt.temp_preview_square);
                const int pad = 10;
                if (raw.cols > inset.cols + 2*pad && raw.rows > inset.rows + 2*pad) {
                    cv::Rect dst(raw.cols - inset.cols - pad, pad, inset.cols, inset.rows);
                    inset.copyTo(raw(dst));
                    cv::rectangle(raw, dst, cv::Scalar(255,255,255), 2);
                    cv::putText(raw, "live warp preview", {dst.x, std::max(18, dst.y-4)}, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255,255,255), 1);
                }
            }
            display = downscale_for_preview(raw, opt.preview_soft_max);
            if (show_help) draw_help_overlay(display, false, selected, move_step, size_step);
        } else {
            if (!apply_warp(frame, locked_pack, warped, &valid)) { err = "failed to apply locked warp"; return false; }
            cv::Mat show = warped.clone();
            draw_rois(show, rois, selected);
            const std::string status = "LOCKED family=" + locked_pack.family + " id=" + std::to_string(locked_pack.id);
            cv::putText(show, status, {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.68, cv::Scalar(0,255,0), 2);
            const cv::Rect rr = roi_to_rect(rois.red_roi, show.size());
            const cv::Rect ir = roi_to_rect(rois.image_roi, show.size());
            const double red_valid = static_cast<double>(cv::countNonZero(valid(rr))) / static_cast<double>(rr.area());
            const double img_valid = static_cast<double>(cv::countNonZero(valid(ir))) / static_cast<double>(ir.area());
            cv::putText(show, "red valid=" + std::to_string(red_valid).substr(0,4) + " image valid=" + std::to_string(img_valid).substr(0,4), {12, 54}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0,255,0), 2);
            display = downscale_for_preview(show, opt.preview_soft_max);
            if (show_help) draw_help_overlay(display, true, selected, move_step, size_step);
        }

        cv::imshow("vision_app", display);
        ++frame_idx;
        const int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') break;
        if (key == 'h') show_help = !show_help;
        if (key == 'u') { locked = false; locker.reset(); locked_pack = {}; temp_preview.release(); }
        if (key == ' ' || key == 13) {
            if (cur.found) {
                if (build_warp_package_from_detection(cur, frame.size(), opt.warp_soft_max, locked_pack, err)) {
                    locked = true; locked_det = cur;
                } else {
                    std::cerr << "Warp build rejected: " << err << "\n"; err.clear();
                }
            }
        }
        if (key == '1') selected = 0;
        if (key == '2') selected = 1;
        if (key == '[') move_step = std::max(0.001, move_step * 0.5);
        if (key == ']') move_step = std::min(0.25, move_step * 2.0);
        if (key == ',') size_step = std::max(0.001, size_step * 0.5);
        if (key == '.') size_step = std::min(0.25, size_step * 2.0);
        if (key == 'r') rois = opt.default_rois;
        if (locked) {
            if (selected == 0) adjust_roi(rois.red_roi, key, move_step, size_step);
            else adjust_roi(rois.image_roi, key, move_step, size_step);
        }
        if (key == 'o') save_rois_yaml(opt.save_rois, rois);
        if (key == 'y' && locked) save_warp_package(opt.save_warp, locked_pack);
        if (key == 'p' && locked) {
            save_rois_yaml(opt.save_rois, rois);
            save_warp_package(opt.save_warp, locked_pack);
            write_report_md(opt.save_report, "Calibration Report", nullptr,
                            std::string("- Locked family: ") + locked_pack.family + "\n" +
                            "- Locked id: " + std::to_string(locked_pack.id) + "\n" +
                            "- Warp size: " + std::to_string(locked_pack.warp_size.width) + "x" + std::to_string(locked_pack.warp_size.height));
        }
    }

    cv::destroyAllWindows();
    return true;
}

inline bool run_deploy(const AppOptions& opt, std::string& err) {
    WarpPackage pack;
    if (!load_warp_package(opt.load_warp, pack)) { err = "failed to load warp package: " + opt.load_warp; return false; }
    RoiConfig rois = opt.default_rois;
    (void)load_rois_yaml(opt.load_rois, rois);

    int cam_w = opt.width, cam_h = opt.height;
    cv::VideoCapture cap;
    if (!open_capture(cap, opt.device, cam_w, cam_h, opt.fps, opt.fourcc, opt.buffer_size, opt.camera_soft_max, err)) return false;

    cv::Mat frame, warped, valid, display;
    cv::namedWindow("vision_app", cv::WINDOW_AUTOSIZE);
    while (true) {
        if (!grab_latest_frame(cap, opt.latest_only, opt.drain_grabs, frame)) { err = "failed to read frame"; return false; }
        if (!apply_warp(frame, pack, warped, &valid)) { err = "failed to apply warp"; return false; }
        cv::Mat show = warped.clone();
        draw_rois(show, rois, -1);
        cv::putText(show, "DEPLOY family=" + pack.family + " id=" + std::to_string(pack.id), {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.68, cv::Scalar(0,255,0), 2);
        display = downscale_for_preview(show, opt.preview_soft_max);
        cv::imshow("vision_app", display);
        const int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') break;
    }
    cv::destroyAllWindows();
    return true;
}

} // namespace vision_app
