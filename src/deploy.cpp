#include "deploy.h"

#include "camera.h"
#include "homography.h"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <ctime>

namespace fs = std::filesystem;

namespace
{
std::string make_stamp()
{
    using clock = std::chrono::system_clock;
    const auto now = clock::now();
    const std::time_t t = clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
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

bool is_red_trigger(const cv::Scalar& mean_bgr, const TriggerConfig& cfg)
{
    const double b = mean_bgr[0];
    const double g = mean_bgr[1];
    const double r = mean_bgr[2];
    return r > cfg.red_threshold && r > g + cfg.red_margin && r > b + cfg.red_margin;
}
}

bool run_deploy(const AppConfig& cfg, std::string* error)
{
    if (!cfg.calibration.valid || cfg.calibration.homography.empty() || cfg.calibration.warped_width <= 0 || cfg.calibration.warped_height <= 0)
    {
        if (error) *error = "Deploy requires a valid saved calibration";
        return false;
    }
    if (!is_valid_ratio_roi(cfg.calibration.red_roi) || !is_valid_ratio_roi(cfg.calibration.image_roi))
    {
        if (error) *error = "Deploy requires both red_roi and image_roi";
        return false;
    }

    cv::VideoCapture cap;
    if (!open_camera(cap, cfg.camera, error))
        return false;

    const cv::Size warped_size(cfg.calibration.warped_width, cfg.calibration.warped_height);
    const cv::Rect red_rect = ratio_to_rect_clamped(cfg.calibration.red_roi, warped_size);
    const cv::Rect image_rect = ratio_to_rect_clamped(cfg.calibration.image_roi, warped_size);

    if (red_rect.width <= 0 || red_rect.height <= 0 || image_rect.width <= 0 || image_rect.height <= 0)
    {
        if (error) *error = "Reconstructed ROI is invalid after clamping";
        return false;
    }

    if (cfg.trigger.save_raw || cfg.trigger.save_warped || cfg.trigger.save_roi)
        fs::create_directories(cfg.trigger.capture_dir);

    const bool show_ui = cfg.runtime.show_ui && !cfg.runtime.headless_deploy;
    if (show_ui)
        cv::namedWindow("deploy", cv::WINDOW_NORMAL);

    cv::Mat frame;
    cv::Mat warped;
    auto last_trigger = std::chrono::steady_clock::time_point{};

    while (true)
    {
        if (!read_frame(cap, frame, cfg.camera.drop_frames_per_read))
        {
            if (error) *error = "Deploy read failed";
            return false;
        }

        if (cfg.camera.flip_horizontal)
            cv::flip(frame, frame, 1);

        if (!warp_frame(frame, warped, cfg.calibration.homography, warped_size))
        {
            if (error) *error = "Warp failed during deploy";
            return false;
        }

        const cv::Mat red_view = warped(red_rect);
        const cv::Scalar mean_bgr = cv::mean(red_view);
        const bool trigger_now = is_red_trigger(mean_bgr, cfg.trigger);
        const auto now = std::chrono::steady_clock::now();
        const bool cooldown_ok = last_trigger.time_since_epoch().count() == 0 ||
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_trigger).count() >= cfg.trigger.cooldown_ms;

        if (trigger_now && cooldown_ok)
        {
            last_trigger = now;
            const cv::Mat roi_view = warped(image_rect);
            const std::string stamp = make_stamp();
            if (cfg.trigger.save_raw)
                cv::imwrite((fs::path(cfg.trigger.capture_dir) / (stamp + "_raw.png")).string(), frame);
            if (cfg.trigger.save_warped)
                cv::imwrite((fs::path(cfg.trigger.capture_dir) / (stamp + "_warped.png")).string(), warped);
            if (cfg.trigger.save_roi)
                cv::imwrite((fs::path(cfg.trigger.capture_dir) / (stamp + "_roi.png")).string(), roi_view.clone());
        }

        if (show_ui)
        {
            cv::Mat vis = warped.clone();
            draw_roi(vis, red_rect, cv::Scalar(0, 0, 255), "red_roi");
            draw_roi(vis, image_rect, cv::Scalar(255, 255, 0), "image_roi");
            std::ostringstream oss;
            oss << "B=" << mean_bgr[0] << " G=" << mean_bgr[1] << " R=" << mean_bgr[2]
                << " trigger=" << (trigger_now ? "1" : "0");
            cv::putText(vis,
                        oss.str(),
                        cv::Point(10, 24),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.6,
                        cv::Scalar(0, 255, 255),
                        2,
                        cv::LINE_AA);
            cv::imshow("deploy", vis);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 27)
                break;
        }
    }

    if (show_ui)
        cv::destroyAllWindows();
    return true;
}
