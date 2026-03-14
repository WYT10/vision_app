#include "app_calib.hpp"
#include "app_camera.hpp"
#include <opencv2/aruco.hpp>
#include <iostream>
#include <map>

cv::Rect2f normalize_rect(const cv::Rect &rect, int img_width, int img_height)
{
    return cv::Rect2f(
        (float)rect.x / img_width, (float)rect.y / img_height,
        (float)rect.width / img_width, (float)rect.height / img_height);
}

void run_calibration(AppConfig &config)
{
    AppCamera cam;
    if (!cam.open({config.camera_id, config.width, config.height, config.fps}))
    {
        std::cerr << "Camera failed to open.\n";
        return;
    }

    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);

    // Initial virtual square mapping (arbitrary size, will be adapted)
    const int TAG_DIST = 400;
    std::vector<cv::Point2f> dst_pts = {
        cv::Point2f(0, 0), cv::Point2f(TAG_DIST, 0),
        cv::Point2f(TAG_DIST, TAG_DIST), cv::Point2f(0, TAG_DIST)};

    cv::Mat frame, bev_frame;
    bool locked = false;

    std::cout << "\n=== STEP 1: Live Adaptive Homography ===\n";
    std::cout << "Place tags 0(TL), 1(TR), 2(BR), 3(BL).\nPress ENTER to lock, 'Q' to quit.\n";

    while (true)
    {
        cam.read(frame);
        if (frame.empty())
            continue;

        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners, rejected;
        detector.detectMarkers(frame, markerCorners, markerIds, rejected);

        if (!markerIds.empty())
            cv::aruco::drawDetectedMarkers(frame, markerCorners, markerIds);

        std::map<int, cv::Point2f> ordered_centers;
        for (size_t i = 0; i < markerIds.size(); i++)
        {
            if (markerIds[i] >= 0 && markerIds[i] <= 3)
            {
                cv::Point2f c(0, 0);
                for (int j = 0; j < 4; j++)
                    c += markerCorners[i][j];
                c.x /= 4.0f;
                c.y /= 4.0f;
                ordered_centers[markerIds[i]] = c;
            }
        }

        if (ordered_centers.size() == 4)
        {
            std::vector<cv::Point2f> src_pts = {
                ordered_centers[0], ordered_centers[1], ordered_centers[2], ordered_centers[3]};

            // 1. Initial Homography
            cv::Mat H = cv::findHomography(src_pts, dst_pts);

            // 2. Find warped bounds of the original camera frame
            std::vector<cv::Point2f> frame_corners = {
                {0, 0}, {(float)frame.cols, 0}, {(float)frame.cols, (float)frame.rows}, {0, (float)frame.rows}};
            std::vector<cv::Point2f> warped_corners;
            cv::perspectiveTransform(frame_corners, warped_corners, H);

            cv::Rect bbox = cv::boundingRect(warped_corners);

            // Safety limit for extreme perspectives
            if (bbox.width > 3000 || bbox.height > 3000)
            {
                std::cout << "\r[Warning] Perspective too extreme, moving tags closer.";
                continue;
            }

            // 3. Translation matrix to shift the bounding box to (0,0)
            cv::Mat T = (cv::Mat_<double>(3, 3) << 1.0, 0.0, -bbox.x, 0.0, 1.0, -bbox.y, 0.0, 0.0, 1.0);

            // 4. Final Adaptive Homography
            config.homography = T * H;
            config.warp_width = bbox.width;
            config.warp_height = bbox.height;

            cv::warpPerspective(frame, bev_frame, config.homography, cv::Size(config.warp_width, config.warp_height));
            cv::imshow("Adaptive BEV (LIVE)", bev_frame);
        }

        cv::imshow("Camera Feed", frame);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q')
            return;
        if (key == 13 && ordered_centers.size() == 4 && !bev_frame.empty())
        { // ENTER
            locked = true;
            break;
        }
    }

    cv::destroyAllWindows();
    if (!locked)
        return;

    // --- STEP 2 & 3: ROI Selection ---
    std::cout << "\n=== STEP 2: Red Line Trigger ROI ===\n";
    cv::Rect px_red = cv::selectROI("Select RED LINE ROI", bev_frame, true, false);
    config.red_line_roi = normalize_rect(px_red, bev_frame.cols, bev_frame.rows);
    cv::destroyWindow("Select RED LINE ROI");

    std::cout << "\n=== STEP 3: YOLO Inference ROI ===\n";
    cv::Rect px_yolo = cv::selectROI("Select YOLO ROI", bev_frame, true, false);
    config.yolo_roi = normalize_rect(px_yolo, bev_frame.cols, bev_frame.rows);
    cv::destroyWindow("Select YOLO ROI");

    config.save("config.yaml");
    std::cout << "\n[Done] Adaptive bounds computed: " << config.warp_width << "x" << config.warp_height << "\n";
}