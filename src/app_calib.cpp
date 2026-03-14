#include "app_calib.hpp"
#include "app_camera.hpp"
#include <opencv2/aruco.hpp>
#include <iostream>
#include <map>

cv::Rect2f norm_rect(const cv::Rect &r, int w, int h)
{
    return {(float)r.x / w, (float)r.y / h, (float)r.width / w, (float)r.height / h};
}

void run_calib_mode(AppConfig &config)
{
    AppCamera cam;
    if (!cam.open(config))
        return;

    cv::aruco::DetectorParameters params = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);
    cv::aruco::ArucoDetector detector(dict, params);

    cv::Mat frame, bev_frame;
    bool locked = false;

    // --- PHASE 1: Adaptive Homography ---
    std::cout << "Place Tags 0(TL), 1(TR), 2(BR), 3(BL). Press ENTER to lock.\n";
    std::vector<cv::Point2f> dst_pts = {{0, 0}, {400, 0}, {400, 400}, {0, 400}};

    while (true)
    {
        cam.read(frame);
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners, rejected;
        detector.detectMarkers(frame, corners, ids, rejected);

        if (!ids.empty())
            cv::aruco::drawDetectedMarkers(frame, corners, ids);

        std::map<int, cv::Point2f> centers;
        for (size_t i = 0; i < ids.size(); i++)
        {
            if (ids[i] >= 0 && ids[i] <= 3)
            {
                cv::Point2f c(0, 0);
                for (int j = 0; j < 4; j++)
                    c += corners[i][j];
                centers[ids[i]] = c / 4.0f;
            }
        }

        if (centers.size() == 4)
        {
            std::vector<cv::Point2f> src = {centers[0], centers[1], centers[2], centers[3]};
            cv::Mat H = cv::findHomography(src, dst_pts);

            std::vector<cv::Point2f> cam_corners = {{0, 0}, {(float)frame.cols, 0}, {(float)frame.cols, (float)frame.rows}, {0, (float)frame.rows}};
            std::vector<cv::Point2f> warped_corners;
            cv::perspectiveTransform(cam_corners, warped_corners, H);
            cv::Rect bbox = cv::boundingRect(warped_corners);

            if (bbox.width < 3000 && bbox.height < 3000)
            {
                cv::Mat T = (cv::Mat_<double>(3, 3) << 1, 0, -bbox.x, 0, 1, -bbox.y, 0, 0, 1);
                config.homography = T * H;
                config.warp_width = bbox.width;
                config.warp_height = bbox.height;

                cv::warpPerspective(frame, bev_frame, config.homography, {config.warp_width, config.warp_height});

                // Draw Alignment Grid
                for (int i = 0; i < bev_frame.cols; i += 50)
                    cv::line(bev_frame, {i, 0}, {i, bev_frame.rows}, {0, 255, 0}, 1);
                for (int i = 0; i < bev_frame.rows; i += 50)
                    cv::line(bev_frame, {0, i}, {bev_frame.cols, i}, {0, 255, 0}, 1);

                cv::imshow("BEV Live", bev_frame);
            }
        }
        cv::imshow("Raw", frame);
        int key = cv::waitKey(1);
        if (key == 13 && !bev_frame.empty())
        {
            locked = true;
            break;
        }
    }
    cv::destroyAllWindows();
    if (!locked)
        return;

    // Remove grid for ROI selection by re-warping clean frame
    cv::warpPerspective(frame, bev_frame, config.homography, {config.warp_width, config.warp_height});

    // --- PHASE 2: ROIs ---
    config.red_line_roi = norm_rect(cv::selectROI("Select Red Line Trigger", bev_frame, true, false), bev_frame.cols, bev_frame.rows);
    config.yolo_roi = norm_rect(cv::selectROI("Select YOLO Input", bev_frame, true, false), bev_frame.cols, bev_frame.rows);
    cv::destroyAllWindows();

    // --- PHASE 3: HSV Tuning ---
    cv::namedWindow("HSV Tuner");
    cv::createTrackbar("H Min", "HSV Tuner", &config.h_min, 179);
    cv::createTrackbar("H Max", "HSV Tuner", &config.h_max, 179);
    cv::createTrackbar("S Min", "HSV Tuner", &config.s_min, 255);
    cv::createTrackbar("S Max", "HSV Tuner", &config.s_max, 255);
    cv::createTrackbar("V Min", "HSV Tuner", &config.v_min, 255);
    cv::createTrackbar("V Max", "HSV Tuner", &config.v_max, 255);

    cv::Rect px_red_roi(config.red_line_roi.x * bev_frame.cols, config.red_line_roi.y * bev_frame.rows,
                        config.red_line_roi.width * bev_frame.cols, config.red_line_roi.height * bev_frame.rows);
    cv::Mat trigger_crop = bev_frame(px_red_roi);
    cv::Mat hsv, mask;

    std::cout << "Tune HSV values for the Red Line. Press ENTER to save and finish.\n";
    while (true)
    {
        cv::cvtColor(trigger_crop, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(config.h_min, config.s_min, config.v_min),
                    cv::Scalar(config.h_max, config.s_max, config.v_max), mask);
        cv::imshow("HSV Tuner", mask);
        if (cv::waitKey(30) == 13)
            break;
    }
    cv::destroyAllWindows();
    config.save("config.yaml");
    std::cout << "Calibration saved to config.yaml\n";
}