#include "camera_test.hpp"

#include <chrono>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

namespace vision_app
{

    static int fourcc_from_string(const std::string &s)
    {
        if (s.size() != 4)
            return 0;
        return cv::VideoWriter::fourcc(s[0], s[1], s[2], s[3]);
    }

    bool run_camera_test(const AppOptions &opt, RuntimeStats &stats, std::string &err)
    {
        stats = {};

        cv::VideoCapture cap(opt.device, cv::CAP_V4L2);
        if (!cap.isOpened())
        {
            err = "cannot open camera: " + opt.device;
            return false;
        }

        if (!opt.fourcc.empty())
        {
            cap.set(cv::CAP_PROP_FOURCC, fourcc_from_string(opt.fourcc));
        }
        cap.set(cv::CAP_PROP_FRAME_WIDTH, opt.width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, opt.height);
        cap.set(cv::CAP_PROP_FPS, opt.fps);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

        cv::Mat frame;
        for (int i = 0; i < opt.warmup_frames; ++i)
        {
            cap.read(frame);
        }

        if (!opt.headless && opt.show_preview)
        {
            cv::namedWindow("vision_app", cv::WINDOW_AUTOSIZE);
        }

        using clk = std::chrono::steady_clock;
        const auto t0 = clk::now();
        auto last = t0;

        double dt_min_ms = std::numeric_limits<double>::max();
        double dt_max_ms = 0.0;
        double dt_sum_ms = 0.0;

        while (true)
        {
            if (!cap.read(frame) || frame.empty())
            {
                err = "failed to read frame";
                return false;
            }

            const auto now = clk::now();
            const double dt_ms = std::chrono::duration<double, std::milli>(now - last).count();
            last = now;

            if (stats.frames > 0)
            {
                if (dt_ms < dt_min_ms)
                    dt_min_ms = dt_ms;
                if (dt_ms > dt_max_ms)
                    dt_max_ms = dt_ms;
                dt_sum_ms += dt_ms;
            }

            ++stats.frames;

            if (!opt.headless && opt.show_preview)
            {
                cv::imshow("vision_app", frame);
                const int key = cv::waitKey(1) & 0xFF;
                if (key == 27 || key == 'q')
                    break;
            }

            const double elapsed_sec = std::chrono::duration<double>(now - t0).count();
            if (elapsed_sec >= opt.duration_sec)
                break;
        }

        const auto t1 = clk::now();
        stats.elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
        stats.width = static_cast<double>(frame.cols);
        stats.height = static_cast<double>(frame.rows);

        if (stats.elapsed_sec > 0.0)
        {
            stats.fps_avg = static_cast<double>(stats.frames) / stats.elapsed_sec;
        }

        if (stats.frames > 1)
        {
            const double avg_ms = dt_sum_ms / static_cast<double>(stats.frames - 1);
            stats.frame_time_avg_ms = avg_ms;
            stats.frame_time_min_ms = dt_min_ms;
            stats.frame_time_max_ms = dt_max_ms;

            if (dt_max_ms > 0.0)
                stats.fps_min = 1000.0 / dt_max_ms;
            if (dt_min_ms > 0.0)
                stats.fps_max = 1000.0 / dt_min_ms;
        }

        cap.release();
        if (!opt.headless && opt.show_preview)
        {
            cv::destroyWindow("vision_app");
        }

        return true;
    }

} // namespace vision_app