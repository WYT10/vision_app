#include "calibration.h"

#include "camera.h"

#include <algorithm>
#include "homography.h"

#include <cctype>
#include <cmath>
#include <map>
#include <opencv2/aruco.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <utility>
#include <vector>

namespace
{
struct Detection
{
    std::string family;
    int id = -1;
    std::vector<cv::Point2f> corners;
};

std::map<std::string, int> make_family_map()
{
    return {
        {"16H5", cv::aruco::DICT_APRILTAG_16h5},
        {"25H9", cv::aruco::DICT_APRILTAG_25h9},
        {"36H10", cv::aruco::DICT_APRILTAG_36h10},
        {"36H11", cv::aruco::DICT_APRILTAG_36h11}
    };
}

std::vector<std::pair<std::string, int>> active_families(const TagConfig& cfg)
{
    const auto fams = make_family_map();
    std::vector<std::pair<std::string, int>> out;

    if (cfg.family_mode == "auto")
    {
        for (const auto& kv : fams)
            out.push_back(kv);
        return out;
    }

    std::string key = cfg.allowed_family;
    for (char& ch : key)
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));

    auto it = fams.find(key);
    if (it != fams.end())
        out.push_back(*it);
    return out;
}

bool detect_tag(const cv::Mat& gray, const TagConfig& cfg, Detection& detection)
{
    const auto families = active_families(cfg);
    for (const auto& [name, dict_id] : families)
    {
        const auto dict = cv::aruco::getPredefinedDictionary(dict_id);
        cv::aruco::DetectorParameters params;
        cv::aruco::ArucoDetector detector(dict, params);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        detector.detectMarkers(gray, corners, ids);

        for (size_t i = 0; i < ids.size(); ++i)
        {
            if (cfg.allowed_id >= 0 && ids[i] != cfg.allowed_id)
                continue;
            detection.family = name;
            detection.id = ids[i];
            detection.corners = corners[i];
            return true;
        }
    }
    return false;
}

void draw_detection_overlay(cv::Mat& frame, const Detection& det)
{
    if (det.corners.size() != 4)
        return;

    std::vector<std::vector<cv::Point>> poly(1);
    for (const auto& p : det.corners)
        poly[0].push_back(cv::Point(static_cast<int>(std::round(p.x)), static_cast<int>(std::round(p.y))));

    cv::polylines(frame, poly, true, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    cv::putText(frame,
                det.family + " id=" + std::to_string(det.id),
                poly[0][0],
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(0, 255, 0),
                2,
                cv::LINE_AA);
}

void draw_roi(cv::Mat& image, const cv::Rect& rect, const cv::Scalar& color, const std::string& label)
{
    if (rect.width <= 0 || rect.height <= 0)
        return;

    cv::rectangle(image, rect, color, 2);
    cv::putText(image,
                label,
                cv::Point(rect.x, std::max(18, rect.y - 4)),
                cv::FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv::LINE_AA);
}

bool select_roi_from_window(const std::string& window_name, const cv::Mat& image, cv::Rect& out_rect)
{
    const cv::Rect2d r = cv::selectROI(window_name, image, false, false);
    cv::Rect rect(static_cast<int>(std::round(r.x)),
                  static_cast<int>(std::round(r.y)),
                  static_cast<int>(std::round(r.width)),
                  static_cast<int>(std::round(r.height)));
    if (rect.width <= 0 || rect.height <= 0)
        return false;
    out_rect = rect;
    return true;
}
}

bool run_calibration(AppConfig& cfg, const std::string& config_path, std::string* error)
{
    cv::VideoCapture cap;
    CameraMode actual_mode;
    if (!open_camera(cap, cfg.camera, &actual_mode, error))
        return false;

    const std::string raw_window = "raw";
    const std::string warp_window = "warped";
    if (cfg.runtime.show_ui)
    {
        cv::namedWindow(raw_window, cv::WINDOW_NORMAL);
        cv::namedWindow(warp_window, cv::WINDOW_NORMAL);
    }

    bool locked = false;
    cv::Mat preview_H;
    cv::Size preview_size;
    cv::Mat locked_H;
    cv::Size locked_size;
    cv::Rect red_rect;
    cv::Rect image_rect;

    cv::Mat frame;
    cv::Mat gray;
    cv::Mat warped;

    while (true)
    {
        if (!read_frame(cap, frame, cfg.camera.drop_frames_per_read))
        {
            if (error) *error = "Calibration read failed";
            return false;
        }

        if (cfg.camera.flip_horizontal)
            cv::flip(frame, frame, 1);

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        Detection det;
        const bool found = detect_tag(gray, cfg.tag, det);

        cv::Mat raw_vis = frame;
        if (cfg.runtime.show_ui)
            raw_vis = frame.clone();
        if (found && cfg.runtime.show_ui)
            draw_detection_overlay(raw_vis, det);

        bool preview_ok = false;
        if (!locked && found)
        {
            preview_ok = compute_homography_from_tag(
                det.corners,
                frame.size(),
                cfg.tag.tag_size_units,
                cfg.tag.output_padding_units,
                preview_H,
                preview_size);
        }

        if (cfg.tag.lock_on_first_detection && !locked && preview_ok)
        {
            locked = true;
            locked_H = preview_H.clone();
            locked_size = preview_size;
        }

        const cv::Mat& H_use = locked ? locked_H : preview_H;
        const cv::Size size_use = locked ? locked_size : preview_size;
        const bool have_preview = !H_use.empty() && size_use.width > 0 && size_use.height > 0;

        if (have_preview)
        {
            if (!warp_frame(frame, warped, H_use, size_use))
            {
                if (error) *error = "Warp failed during calibration";
                return false;
            }

            if (cfg.runtime.show_ui)
            {
                if (red_rect.width > 0 && red_rect.height > 0)
                    draw_roi(warped, red_rect, cv::Scalar(0, 0, 255), "red_roi");
                if (image_rect.width > 0 && image_rect.height > 0)
                    draw_roi(warped, image_rect, cv::Scalar(255, 255, 0), "image_roi");
            }
        }

        if (!cfg.runtime.show_ui)
        {
            if (locked && have_preview)
                break;
            continue;
        }

        const std::string status = locked ? "LOCKED" : "LIVE";
        cv::putText(raw_vis,
                    "[L] lock  [R] red ROI  [I] image ROI  [S] save  [ESC] exit   status=" + status,
                    cv::Point(10, 24),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.55,
                    cv::Scalar(0, 255, 255),
                    2,
                    cv::LINE_AA);
        cv::imshow(raw_window, raw_vis);
        if (have_preview)
            cv::imshow(warp_window, warped);

        const int key = cv::waitKey(1) & 0xFF;
        if (key == 27)
            break;
        if (key == 'l' || key == 'L')
        {
            if (preview_ok)
            {
                locked = !locked;
                if (locked)
                {
                    locked_H = preview_H.clone();
                    locked_size = preview_size;
                    red_rect = {};
                    image_rect = {};
                    cfg.calibration.red_roi_ratio = {};
                    cfg.calibration.image_roi_ratio = {};
                }
            }
        }
        else if (key == 'r' || key == 'R')
        {
            if (locked && have_preview)
            {
                cv::Mat frozen = warped.clone();
                if (select_roi_from_window(warp_window, frozen, red_rect))
                    cfg.calibration.red_roi_ratio = rect_to_ratio(red_rect, locked_size);
            }
        }
        else if (key == 'i' || key == 'I')
        {
            if (locked && have_preview)
            {
                cv::Mat frozen = warped.clone();
                if (select_roi_from_window(warp_window, frozen, image_rect))
                    cfg.calibration.image_roi_ratio = rect_to_ratio(image_rect, locked_size);
            }
        }
        else if (key == 's' || key == 'S')
        {
            if (!locked || locked_H.empty() || !is_valid_ratio_roi(cfg.calibration.red_roi_ratio) || !is_valid_ratio_roi(cfg.calibration.image_roi_ratio))
            {
                if (error) *error = "Need locked homography + both ROIs before save";
                continue;
            }

            cfg.calibration.valid = true;
            cfg.calibration.camera_mode_used = actual_mode;
            cfg.calibration.homography = locked_H.clone();
            cfg.calibration.warped_width = locked_size.width;
            cfg.calibration.warped_height = locked_size.height;

            std::string save_error;
            if (!save_config(config_path, cfg, &save_error))
            {
                if (error) *error = save_error;
                return false;
            }
        }
    }

    if (cfg.runtime.show_ui)
        cv::destroyAllWindows();
    return true;
}
