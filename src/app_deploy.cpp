#include "app_deploy.hpp"
#include "app_camera.hpp"
#include <iostream>

// Zero-Waste Map Generator Helper
void generate_roi_maps(const AppConfig &cfg, const cv::Rect2f &norm_roi, cv::Mat &map_x_16, cv::Mat &map_y_16)
{
    int roi_x = norm_roi.x * cfg.warp_width;
    int roi_y = norm_roi.y * cfg.warp_height;
    int roi_w = norm_roi.width * cfg.warp_width;
    int roi_h = norm_roi.height * cfg.warp_height;

    cv::Mat H_inv = cfg.homography.inv();
    cv::Mat map_x(roi_h, roi_w, CV_32FC1);
    cv::Mat map_y(roi_h, roi_w, CV_32FC1);

    for (int r = 0; r < roi_h; ++r)
    {
        for (int c = 0; c < roi_w; ++c)
        {
            double px = c + roi_x;
            double py = r + roi_y;
            double Z = H_inv.at<double>(2, 0) * px + H_inv.at<double>(2, 1) * py + H_inv.at<double>(2, 2);
            double X = (H_inv.at<double>(0, 0) * px + H_inv.at<double>(0, 1) * py + H_inv.at<double>(0, 2)) / Z;
            double Y = (H_inv.at<double>(1, 0) * px + H_inv.at<double>(1, 1) * py + H_inv.at<double>(1, 2)) / Z;
            map_x.at<float>(r, c) = (float)X;
            map_y.at<float>(r, c) = (float)Y;
        }
    }
    cv::convertMaps(map_x, map_y, map_x_16, map_y_16, CV_16SC2);
}

void run_deploy_mode(const AppConfig &config, bool no_ui)
{
    if (config.homography.empty())
    {
        std::cerr << "Run --mode calib first!\n";
        return;
    }

    AppCamera cam;
    if (!cam.open(config))
        return;

    // Startup Pre-computation (Fast NEON 16-bit maps)
    cv::Mat red_map_x, red_map_y, yolo_map_x, yolo_map_y;
    generate_roi_maps(config, config.red_line_roi, red_map_x, red_map_y);
    generate_roi_maps(config, config.yolo_roi, yolo_map_x, yolo_map_y);

    cv::Mat frame, red_crop, hsv, mask, yolo_crop;
    const int RED_THRESHOLD = 50; // Minimum pixels to trigger

    std::cout << "Deploy Loop Running...\n";

    while (true)
    {
        if (!cam.read(frame) || frame.empty())
            continue;

        // 1. Fast Gate: Remap ONLY the Red Line ROI
        cv::remap(frame, red_crop, red_map_x, red_map_y, cv::INTER_LINEAR);
        cv::cvtColor(red_crop, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(config.h_min, config.s_min, config.v_min),
                    cv::Scalar(config.h_max, config.s_max, config.v_max), mask);

        int red_pixels = cv::countNonZero(mask);

        // 2. Heavy Gate: Trigger ONNX/NCNN only if red line detected
        if (red_pixels > RED_THRESHOLD)
        {
            cv::remap(frame, yolo_crop, yolo_map_x, yolo_map_y, cv::INTER_LINEAR);

            // TODO: Pass `yolo_crop` to your ModelRunner here

            if (!no_ui)
                cv::imshow("YOLO Input", yolo_crop);
        }
        else
        {
            if (!no_ui)
                cv::destroyWindow("YOLO Input");
        }

        if (!no_ui)
        {
            cv::imshow("Red Mask", mask);
            if (cv::waitKey(1) == 'q')
                break;
        }
    }
}