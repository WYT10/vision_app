#include <iostream>
#include <opencv2/opencv.hpp>
#include "app_config.hpp"
#include "app_test.hpp"
#include "app_calib.hpp"
#include "app_deploy.hpp"

int main(int argc, char **argv)
{
    const std::string keys =
        "{help h    |       | print usage }"
        "{mode      |test   | app mode: test, calib, deploy }"
        "{camera    |0      | camera device id }"
        "{width     |640    | requested width }"
        "{height    |480    | requested height }"
        "{fps       |60     | requested fps }"
        "{no-ui     |false  | run entirely headless }";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    std::string mode = parser.get<std::string>("mode");
    bool no_ui = parser.get<bool>("no-ui");

    AppConfig config;
    config.load("config.yaml");

    if (parser.has("camera"))
        config.camera_id = parser.get<int>("camera");
    if (parser.has("width"))
        config.width = parser.get<int>("width");
    if (parser.has("height"))
        config.height = parser.get<int>("height");
    if (parser.has("fps"))
        config.fps = parser.get<int>("fps");

    if (mode == "test")
    {
        run_test_mode(config, no_ui);
    }
    else if (mode == "calib")
    {
        if (no_ui)
        {
            std::cerr << "Calibration requires a UI.\n";
            return -1;
        }
        run_calib_mode(config);
    }
    else if (mode == "deploy")
    {
        run_deploy_mode(config, no_ui);
    }
    else
    {
        std::cerr << "Unknown mode.\n";
    }
    return 0;
}