#include "deploy.hpp"

#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "camera.hpp"
#include "constants.hpp"
#include "utils.hpp"
#include "warp.hpp"

namespace app {

static double computeRedRatio(const cv::Mat& roi_bgr, const RedThreshold& red, int& mean_r_out) {
    cv::Mat hsv;
    cv::cvtColor(roi_bgr, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask1, mask2, mask;
    cv::inRange(hsv, cv::Scalar(red.hue_low_1, red.sat_min, red.val_min), cv::Scalar(red.hue_high_1, 255, 255), mask1);
    cv::inRange(hsv, cv::Scalar(red.hue_low_2, red.sat_min, red.val_min), cv::Scalar(red.hue_high_2, 255, 255), mask2);
    cv::bitwise_or(mask1, mask2, mask);

    const double ratio = static_cast<double>(cv::countNonZero(mask)) / static_cast<double>(mask.total());
    mean_r_out = static_cast<int>(cv::mean(roi_bgr)[2]);
    return ratio;
}

int runDeploy(const AppConfig& config) {
    if (!config.calibration.valid) {
        std::cerr << "Calibration not valid. Run calibrate first.\n";
        return 1;
    }

    auto capture = openCamera(config.camera);
    const cv::Mat H = calibrationHomographyToMat(config.calibration);
    const cv::Size out_size(config.calibration.warp_width, config.calibration.warp_height);

    if (config.deploy.show_windows) {
        cv::namedWindow(kWindowDeployRaw, cv::WINDOW_NORMAL);
        cv::namedWindow(kWindowDeployWarp, cv::WINDOW_NORMAL);
    }

    int cooldown = 0;
    ensureDir(config.deploy.save_dir);

    for (;;) {
        cv::Mat frame = readFrame(capture, config.camera.flip_horizontal);
        if (frame.empty()) break;

        cv::Mat warped = warpFrame(frame, H, out_size);
        const cv::Rect red_rect = denormalizeRoi(config.calibration.red_roi, warped.cols, warped.rows);
        const cv::Rect image_rect = denormalizeRoi(config.calibration.image_roi, warped.cols, warped.rows);

        int mean_r = 0;
        const double red_ratio = computeRedRatio(warped(red_rect), config.red, mean_r);
        const bool triggered = cooldown == 0 && red_ratio >= config.red.ratio_trigger && mean_r >= config.red.pixel_mean_r_min;

        if (triggered) {
            cooldown = config.red.cooldown_frames;
            std::cout << "Triggered: red_ratio=" << red_ratio << ", mean_r=" << mean_r << '\n';

            if (config.deploy.save_trigger_images) {
                const std::string stamp = nowStamp();
                cv::imwrite(config.deploy.save_dir + "/raw_" + stamp + ".png", frame);
                cv::imwrite(config.deploy.save_dir + "/warp_" + stamp + ".png", warped);
                cv::imwrite(config.deploy.save_dir + "/image_roi_" + stamp + ".png", warped(image_rect));
            }
        }
        cooldown = std::max(0, cooldown - 1);

        if (config.deploy.draw_debug) {
            cv::rectangle(warped, red_rect, triggered ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 255), 2);
            cv::rectangle(warped, image_rect, cv::Scalar(255, 255, 0), 2);
            cv::putText(warped, "red_ratio=" + std::to_string(red_ratio).substr(0, 5), {20, 30},
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2);
        }

        if (config.deploy.show_windows) {
            cv::imshow(kWindowDeployRaw, frame);
            cv::imshow(kWindowDeployWarp, warped);
            const int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}

} // namespace app
