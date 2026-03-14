#include "app_test.hpp"
#include "app_camera.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

void run_test_mode(const AppConfig &config, bool no_ui)
{
    AppCamera cam;
    if (!cam.open(config))
    {
        std::cerr << "[Error] Camera failed to open.\n";
        return;
    }

    cv::Mat frame;
    int frame_count = 0;
    double fps = 0.0;
    auto timer_start = std::chrono::high_resolution_clock::now();

    std::cout << "Starting Test Mode. " << (no_ui ? "(Headless)" : "(Visual)") << "\n";

    while (true)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        if (!cam.read(frame) || frame.empty())
            continue;
        auto t1 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> latency = t1 - t0;
        frame_count++;

        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = now - timer_start;

        if (elapsed.count() >= 1.0)
        {
            fps = frame_count / elapsed.count();
            if (no_ui)
            {
                std::cout << "\rFPS: " << std::fixed << std::setprecision(1) << fps
                          << " | Latency: " << std::setprecision(2) << latency.count() << "ms" << std::flush;
            }
            frame_count = 0;
            timer_start = now;
        }

        if (!no_ui)
        {
            cv::putText(frame, "FPS: " + std::to_string((int)fps), {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 1, {0, 255, 0}, 2);
            cv::imshow("Hardware Test", frame);
            if (cv::waitKey(1) == 'q')
                break;
        }
    }
}