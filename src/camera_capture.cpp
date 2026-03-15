#include "camera_capture.hpp"

#include <chrono>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

namespace vision_app {

static int fourcc_from_string(const std::string& s) {
    if (s.size() != 4) return 0;
    return cv::VideoWriter::fourcc(s[0], s[1], s[2], s[3]);
}

static int backend_from_option(const std::string& api) {
    if (api == "v4l2") return cv::CAP_V4L2;
    return cv::CAP_ANY;
}

static bool capture_one(cv::VideoCapture& cap, const AppOptions& opt, cv::Mat& frame, uint64_t& discarded, uint64_t& empty_frames) {
    if (opt.io_mode == IoMode::Read && !opt.latest_only) {
        if (!cap.read(frame) || frame.empty()) {
            ++empty_frames;
            return false;
        }
        return true;
    }

    int grabs = 1;
    if (opt.latest_only && opt.drain_grabs > 1) grabs = opt.drain_grabs;

    for (int i = 0; i < grabs; ++i) {
        if (!cap.grab()) {
            ++empty_frames;
            return false;
        }
        if (i + 1 < grabs) ++discarded;
    }

    if (!cap.retrieve(frame) || frame.empty()) {
        ++empty_frames;
        return false;
    }
    return true;
}

bool run_camera_test(const AppOptions& opt, RuntimeStats& stats, std::string& err) {
    stats = {};
    stats.requested_fps = static_cast<double>(opt.fps);

    cv::VideoCapture cap(opt.device, backend_from_option(opt.capture_api));
    if (!cap.isOpened()) {
        err = "cannot open camera: " + opt.device;
        return false;
    }

    if (!opt.fourcc.empty()) {
        cap.set(cv::CAP_PROP_FOURCC, fourcc_from_string(opt.fourcc));
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, opt.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, opt.height);
    cap.set(cv::CAP_PROP_FPS, opt.fps);

    stats.backend_buffer_request_ok = cap.set(cv::CAP_PROP_BUFFERSIZE, opt.buffer_size);
    stats.backend_buffer_size_after_set = cap.get(cv::CAP_PROP_BUFFERSIZE);

    cv::Mat frame;
    for (int i = 0; i < opt.warmup_frames; ++i) {
        capture_one(cap, opt, frame, stats.stale_grabs_discarded, stats.empty_frames);
    }

    if (!opt.headless && opt.show_preview) {
        cv::namedWindow("vision_app", cv::WINDOW_AUTOSIZE);
    }

    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    auto last = t0;

    double dt_min_ms = std::numeric_limits<double>::max();
    double dt_max_ms = 0.0;
    double dt_sum_ms = 0.0;

    while (true) {
        if (!capture_one(cap, opt, frame, stats.stale_grabs_discarded, stats.empty_frames)) {
            err = "failed to capture frame";
            return false;
        }

        const auto now = clk::now();
        const double dt_ms = std::chrono::duration<double, std::milli>(now - last).count();
        last = now;

        if (stats.frames > 0) {
            if (dt_ms < dt_min_ms) dt_min_ms = dt_ms;
            if (dt_ms > dt_max_ms) dt_max_ms = dt_ms;
            dt_sum_ms += dt_ms;
        }

        ++stats.frames;

        if (!opt.headless && opt.show_preview) {
            cv::imshow("vision_app", frame);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q') break;
        }

        const double elapsed_sec = std::chrono::duration<double>(now - t0).count();
        if (elapsed_sec >= opt.duration_sec) break;
    }

    const auto t1 = clk::now();
    stats.elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
    stats.actual_width = static_cast<double>(frame.cols);
    stats.actual_height = static_cast<double>(frame.rows);

    if (stats.elapsed_sec > 0.0) {
        stats.fps_avg = static_cast<double>(stats.frames) / stats.elapsed_sec;
    }
    if (stats.frames > 1) {
        stats.frame_time_avg_ms = dt_sum_ms / static_cast<double>(stats.frames - 1);
        stats.frame_time_min_ms = dt_min_ms;
        stats.frame_time_max_ms = dt_max_ms;
        if (dt_max_ms > 0.0) stats.fps_min = 1000.0 / dt_max_ms;
        if (dt_min_ms > 0.0) stats.fps_max = 1000.0 / dt_min_ms;
    }
    if (stats.requested_fps > 0.0) {
        stats.target_ratio = stats.fps_avg / stats.requested_fps;
        stats.target_met = stats.fps_avg + 0.5 >= stats.requested_fps;
    }

    cap.release();
    if (!opt.headless && opt.show_preview) {
        cv::destroyWindow("vision_app");
    }
    return true;
}

} // namespace vision_app
