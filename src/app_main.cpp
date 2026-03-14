#include <iostream>
#include <opencv2/opencv.hpp>
#include "app_config.hpp"
#include "app_camera.hpp"
#include "app_calib.hpp"

void run_test(AppConfig &config)
{
    AppCamera cam;
    if (!cam.open({config.camera_id, config.width, config.height, config.fps}))
        return;

    cv::Mat frame;
    std::cout << "Running Test Mode. Press 'Q' to quit.\n";
    while (true)
    {
        cam.read(frame);
        if (!frame.empty())
            cv::imshow("Test Mode", frame);
        if (cv::waitKey(1) == 'q')
            break;
    }
}

int main(int argc, char **argv)
{
    const std::string keys =
        "{help h    |       | print usage }"
        "{mode      |test   | app mode: test, calib, deploy }"
        "{camera    |0      | camera device id }"
        "{width     |640    | requested width }"
        "{height    |480    | requested height }"
        "{fps       |60     | requested fps }";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    std::string mode = parser.get<std::string>("mode");

    AppConfig config;
    // Attempt to load existing config, overwrite with CLI flags
    config.load("config.yaml");
    config.camera_id = parser.get<int>("camera");
    config.width = parser.get<int>("width");
    config.height = parser.get<int>("height");
    config.fps = parser.get<int>("fps");

    if (mode == "test")
    {
        run_test(config);
    }
    else if (mode == "calib")
    {
        run_calibration(config);
    }
    else if (mode == "deploy")
    {
        std::cout << "Deploy mode pending implementation.\n";
    }
    else
    {
        std::cerr << "Unknown mode.\n";
    }

    return 0;
}