#include "probe.hpp"

#include <chrono>
#include <fstream>
#include <numeric>

#include <nlohmann/json.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "camera.hpp"
#include "utils.hpp"

namespace app {
using json = nlohmann::json;

static void writeProbeReports(const std::vector<ProbeResult>& results, const fs::path& report_dir) {
    ensureDir(report_dir);
    const std::string stamp = nowStamp();

    json report = json::array();
    std::ofstream csv(report_dir / ("probe_" + stamp + ".csv"));
    csv << "camera_index,backend,req_w,req_h,req_fps,act_w,act_h,fps_measured,open_ok,read_ok,stable,mean_luma,luma_std,note\n";

    for (const auto& r : results) {
        report.push_back({
            {"camera_index", r.camera_index},
            {"backend", r.backend},
            {"req_w", r.req_w},
            {"req_h", r.req_h},
            {"req_fps", r.req_fps},
            {"act_w", r.act_w},
            {"act_h", r.act_h},
            {"fps_measured", r.fps_measured},
            {"open_ok", r.open_ok},
            {"read_ok", r.read_ok},
            {"stable", r.stable},
            {"mean_luma", r.mean_luma},
            {"luma_std", r.luma_std},
            {"note", r.note}
        });

        csv << r.camera_index << ',' << r.backend << ',' << r.req_w << ',' << r.req_h << ',' << r.req_fps << ','
            << r.act_w << ',' << r.act_h << ',' << r.fps_measured << ',' << r.open_ok << ',' << r.read_ok << ','
            << r.stable << ',' << r.mean_luma << ',' << r.luma_std << ',' << '"' << r.note << '"' << '\n';
    }

    std::ofstream(report_dir / ("probe_" + stamp + ".json")) << report.dump(2) << '\n';
}

ProbeResult probeSingleCombination(int camera_index, int backend, int width, int height, int fps) {
    ProbeResult result;
    result.camera_index = camera_index;
    result.backend = backendName(backend);
    result.req_w = width;
    result.req_h = height;
    result.req_fps = fps;

    cv::VideoCapture capture(camera_index, backend);
    result.open_ok = capture.isOpened();
    if (!result.open_ok) {
        result.note = "open failed";
        return result;
    }

    capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    capture.set(cv::CAP_PROP_FPS, fps);
    capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    result.act_w = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    result.act_h = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

    constexpr int sample_frames = 30;
    std::vector<double> lumas;
    int ok_reads = 0;
    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < sample_frames; ++i) {
        cv::Mat frame;
        capture >> frame;
        if (frame.empty()) continue;
        ++ok_reads;

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Scalar mean, stddev;
        cv::meanStdDev(gray, mean, stddev);
        lumas.push_back(mean[0]);
        result.luma_std += stddev[0];
    }

    auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();

    result.read_ok = ok_reads > 0;
    result.stable = ok_reads > sample_frames * 0.9;
    result.fps_measured = seconds > 0.0 ? static_cast<double>(ok_reads) / seconds : 0.0;
    if (!lumas.empty()) {
        result.mean_luma = std::accumulate(lumas.begin(), lumas.end(), 0.0) / static_cast<double>(lumas.size());
        result.luma_std /= static_cast<double>(lumas.size());
    }
    if (!result.stable) result.note = "unstable reads";
    return result;
}

std::vector<ProbeResult> runProbe(const AppConfig&, const fs::path& report_dir) {
    std::vector<ProbeResult> results;

#ifdef _WIN32
    const std::vector<int> backends{cv::CAP_ANY, cv::CAP_MSMF, cv::CAP_DSHOW};
#else
    const std::vector<int> backends{cv::CAP_ANY, cv::CAP_V4L2, cv::CAP_GSTREAMER};
#endif
    const std::vector<cv::Size> sizes{{640, 480}, {1280, 720}, {1920, 1080}};
    const std::vector<int> fps_targets{30, 60};

    for (int camera = 0; camera <= 4; ++camera) {
        for (int backend : backends) {
            for (const auto& size : sizes) {
                for (int fps : fps_targets) {
                    results.push_back(probeSingleCombination(camera, backend, size.width, size.height, fps));
                }
            }
        }
    }

    writeProbeReports(results, report_dir);
    return results;
}

} // namespace app
