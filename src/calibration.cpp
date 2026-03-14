#include "calibration.h"

#include "camera.h"
#include "homography.h"

#include <algorithm>
#include <map>
#include <utility>
#include <opencv2/aruco.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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
        {"16h5", cv::aruco::DICT_APRILTAG_16h5},
        {"25h9", cv::aruco::DICT_APRILTAG_25h9},
        {"36h10", cv::aruco::DICT_APRILTAG_36h10},
        {"36h11", cv::aruco::DICT_APRILTAG_36h11}
    };
}

std::vector<std::pair<std::string, int>> active_families(const TagConfig& cfg)
{
    const auto fams = make_family_map();
    std::vector<std::pair<std::string, int>> out;
    if (cfg.family == "auto")
    {
        for (const auto& kv : fams)
            out.push_back(kv);
        return out;
    }

    auto it = fams.find(cfg.family);
    if (it != fams.end())
        out.push_back(*it);
    return out;
}

bool detect_tag(const cv::Mat& gray, const TagConfig& cfg, Detection& detection)
{
    const auto families = active_families(cfg);
    for (const auto& [name, dict_id] : families)
    {
        auto dict = cv::aruco::getPredefinedDictionary(dict_id);
        auto params = cv::aruco::DetectorParameters::create();
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        cv::aruco::detectMarkers(gray, dict, corners, ids, params);

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

void draw_detection(cv::Mat& frame, const Detection& det)
{
    if (det.corners.size() != 4)
        return;
    for (int i = 0; i < 4; ++i)
    {
        const auto& a = det.corners[i];
        const auto& b = det.corners[(i + 1) % 4];
        cv::line(frame, a, b, cv::Scalar(0, 255, 0), 2);
    }
    cv::putText(frame,
                det.family + " id=" + std::to_string(det.id),
                det.corners[0] + cv::Point2f(5.0f, -5.0f),
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
                0.6,
                color,
                2,
                cv::LINE_AA);
}

bool select_roi_from_window(const std::string& window_name, const cv::Mat& image, cv::Rect& out_rect)
{
    const cv::Rect selected = cv::selectROI(window_name, image, false, false);
    if (selected.width <= 0 || selected.height <= 0)
        return false;
    out_rect = selected;
    return true;
}
}

bool run_calibration(AppConfig& cfg, const std::string& config_path, std::string* error)
{
    if (!cfg.runtime.show_ui && !cfg.tag.lock_on_first_detection)
    {
        if (error) *error = "Headless calibration requires lock_on_first_detection=true";
        return false;
    }

    cv::VideoCapture cap;
    if (!open_camera(cap, cfg.camera, error))
        return false;

    const std::string raw_window = "raw";
    const std::string warp_window = "warped";
    if (cfg.runtime.show_ui)
    {
        cv::namedWindow(raw_window, cv::WINDOW_NORMAL);
        cv::namedWindow(warp_window, cv::WINDOW_NORMAL);
    }

    cv::Mat frame;
    cv::Mat gray;
    cv::Mat warped;
    cv::Mat locked_H;
    cv::Size locked_size;
    bool locked = false;
    bool have_preview = false;

    cv::Rect red_rect;
    cv::Rect image_rect;

    while (true)
    {
        if (!read_frame(cap, frame, cfg.camera.drop_frames_per_read))
        {
            if (error) *error = "Failed to read frame during calibration";
            return false;
        }

        if (cfg.camera.flip_horizontal)
            cv::flip(frame, frame, 1);

        if (frame.channels() == 3)
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        else
            gray = frame;

        Detection det;
        bool found = detect_tag(gray, cfg.tag, det);

        cv::Mat display_raw = frame;
        if (cfg.runtime.show_ui && found)
            draw_detection(display_raw, det);

        cv::Mat preview_H;
        cv::Size preview_size;
        bool preview_ok = false;
        if (found)
        {
            preview_ok = compute_homography_from_tag(det.corners,
                                                     frame.size(),
                                                     cfg.tag.tag_size_units,
                                                     cfg.tag.output_padding_units,
                                                     preview_H,
                                                     preview_size);
        }

        if (!locked && preview_ok && cfg.tag.lock_on_first_detection)
        {
            locked = true;
            locked_H = preview_H.clone();
            locked_size = preview_size;
            red_rect = {};
            image_rect = {};
            cfg.calibration.red_roi = {};
            cfg.calibration.image_roi = {};
        }

        const cv::Mat& H_use = (locked ? locked_H : preview_H);
        const cv::Size size_use = (locked ? locked_size : preview_size);
        have_preview = !H_use.empty() && size_use.width > 0 && size_use.height > 0;

        if (have_preview)
        {
            warp_frame(frame, warped, H_use, size_use);
            if (cfg.runtime.show_ui)
            {
                if (red_rect.width > 0 && red_rect.height > 0)
                    draw_roi(warped, red_rect, cv::Scalar(0, 0, 255), "red_roi");
                if (image_rect.width > 0 && image_rect.height > 0)
                    draw_roi(warped, image_rect, cv::Scalar(255, 255, 0), "image_roi");
            }
        }

        if (cfg.runtime.show_ui)
        {
            cv::Mat raw_vis = display_raw.clone();
            std::string status = locked ? "LOCKED" : "LIVE";
            cv::putText(raw_vis,
                        "[L] lock/unlock  [R] red ROI  [I] image ROI  [S] save  [ESC] exit   status=" + status,
                        cv::Point(10, 25),
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
                        cfg.calibration.red_roi = {};
                        cfg.calibration.image_roi = {};
                    }
                }
            }
            else if (key == 'r' || key == 'R')
            {
                if (!locked || !have_preview)
                    continue;
                cv::Mat frozen = warped.clone();
                if (select_roi_from_window(warp_window, frozen, red_rect))
                    cfg.calibration.red_roi = rect_to_ratio(red_rect, locked_size);
            }
            else if (key == 'i' || key == 'I')
            {
                if (!locked || !have_preview)
                    continue;
                cv::Mat frozen = warped.clone();
                if (select_roi_from_window(warp_window, frozen, image_rect))
                    cfg.calibration.image_roi = rect_to_ratio(image_rect, locked_size);
            }
            else if (key == 's' || key == 'S')
            {
                if (!locked || locked_H.empty() || !is_valid_ratio_roi(cfg.calibration.red_roi) || !is_valid_ratio_roi(cfg.calibration.image_roi))
                {
                    if (error) *error = "Calibration save blocked: need locked homography + both ROIs";
                    continue;
                }
                cfg.calibration.valid = true;
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
        else
        {
            if (cfg.tag.lock_on_first_detection && !locked && preview_ok)
            {
                locked = true;
                locked_H = preview_H.clone();
                locked_size = preview_size;
            }
            if (locked && have_preview)
                break;
        }
    }

    if (locked && !locked_H.empty())
    {
        cfg.calibration.valid = true;
        cfg.calibration.homography = locked_H.clone();
        cfg.calibration.warped_width = locked_size.width;
        cfg.calibration.warped_height = locked_size.height;

        if (is_valid_ratio_roi(cfg.calibration.red_roi) && is_valid_ratio_roi(cfg.calibration.image_roi))
        {
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
