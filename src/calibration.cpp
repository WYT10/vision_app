#include "calibration.hpp"

#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "camera.hpp"
#include "config.hpp"
#include "constants.hpp"
#include "roi_selector.hpp"
#include "tag_detector.hpp"
#include "warp.hpp"

namespace app {

static void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections, const std::optional<Detection>& selected) {
    for (const auto& d : detections) {
        const bool chosen = selected && d.id == selected->id && d.family == selected->family;
        const cv::Scalar color = chosen ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 128, 255);
        for (size_t i = 0; i < d.corners.size(); ++i) {
            cv::line(frame, d.corners[i], d.corners[(i + 1) % d.corners.size()], color, chosen ? 3 : 2);
        }
        cv::putText(frame, d.family + " id=" + std::to_string(d.id), d.corners[0] + cv::Point2f(4.0f, -8.0f),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2);
    }
}

int runCalibration(AppConfig& config, const fs::path& config_path) {
    auto capture = openCamera(config.camera);

    cv::namedWindow(kWindowCalRaw, cv::WINDOW_NORMAL);
    cv::namedWindow(kWindowCalWarp, cv::WINDOW_NORMAL);

    SelectionState selector;
    attachSelectionMouseCallback(kWindowCalWarp, &selector);

    cv::Mat locked_H;
    cv::Size locked_size;

    std::cout << "Calibration controls:\n"
              << "  l = lock current homography\n"
              << "  r = draw red ROI on warped frame\n"
              << "  i = draw image ROI on warped frame\n"
              << "  s = save config\n"
              << "  q / ESC = quit\n";

    for (;;) {
        cv::Mat frame = readFrame(capture, config.camera.flip_horizontal);
        if (frame.empty()) break;

        const auto detections = detectTags(frame);
        const auto selected = selectDetection(detections, config.tag);

        cv::Mat raw_view = frame.clone();
        cv::Mat warp_view;
        cv::Mat live_H;
        cv::Size live_size;

        if (selected && selected->corners.size() == 4) {
            std::tie(live_H, live_size) = buildHomographyAndSize(selected->corners);
            warp_view = warpFrame(frame, live_H, live_size);
        } else {
            warp_view = cv::Mat(480, 640, CV_8UC3, cv::Scalar(24, 24, 24));
        }

        drawDetections(raw_view, detections, selected);
        drawSelectionOverlay(warp_view, selector);

        if (!locked_H.empty()) {
            cv::putText(raw_view, "LOCKED", {20, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.9, {0, 255, 0}, 2);
            cv::putText(warp_view, "Locked transform loaded", {20, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2);
        }

        cv::imshow(kWindowCalRaw, raw_view);
        cv::imshow(kWindowCalWarp, !locked_H.empty() ? warpFrame(frame, locked_H, locked_size) : warp_view);

        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') break;
        if ((key == 'l' || key == 'L') && !live_H.empty() && selected) {
            locked_H = live_H.clone();
            locked_size = live_size;
            config.calibration.valid = true;
            config.calibration.family = selected->family;
            config.calibration.id = selected->id;
            config.calibration.source_frame_width = frame.cols;
            config.calibration.source_frame_height = frame.rows;
            config.calibration.warp_width = locked_size.width;
            config.calibration.warp_height = locked_size.height;
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    config.calibration.H[r * 3 + c] = locked_H.at<double>(r, c);
                }
            }
            for (int i = 0; i < 4; ++i) {
                config.calibration.tag_corners[i] = {selected->corners[i].x, selected->corners[i].y};
            }
            std::cout << "Homography locked. Warp size = " << locked_size.width << " x " << locked_size.height << '\n';
        }
        if ((key == 'r' || key == 'R') && !locked_H.empty()) {
            beginSelection(selector, &config.calibration.red_roi, locked_size.width, locked_size.height, "red_roi");
            std::cout << "Draw red ROI on the warped frame.\n";
        }
        if ((key == 'i' || key == 'I') && !locked_H.empty()) {
            beginSelection(selector, &config.calibration.image_roi, locked_size.width, locked_size.height, "image_roi");
            std::cout << "Draw image ROI on the warped frame.\n";
        }
        if (key == 's' || key == 'S') {
            saveConfig(config, config_path);
            std::cout << "Config saved to " << config_path << '\n';
        }
    }

    cv::destroyWindow(kWindowCalRaw);
    cv::destroyWindow(kWindowCalWarp);
    return 0;
}

} // namespace app
