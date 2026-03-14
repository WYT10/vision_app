#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ----------------------------- constants -----------------------------------
static constexpr const char* kWindowProbe = "probe";
static constexpr const char* kWindowCalRaw = "calibration_raw";
static constexpr const char* kWindowCalWarp = "calibration_warp";
static constexpr const char* kWindowDeployRaw = "deploy_raw";
static constexpr const char* kWindowDeployWarp = "deploy_warp";

// ----------------------------- small utils ---------------------------------
static std::string nowStamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

static void ensureDir(const fs::path& p) {
    if (!p.empty()) fs::create_directories(p);
}

static double clamp01(double x) { return std::max(0.0, std::min(1.0, x)); }

static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return (char)std::tolower(c); });
    return s;
}

// ----------------------------- data types ----------------------------------
struct RoiNorm {
    double x = 0.25;
    double y = 0.25;
    double w = 0.25;
    double h = 0.25;
};

struct CameraProfile {
    int index = 0;
    int width = 640;
    int height = 480;
    int fps = 30;
    std::string backend = "ANY";
    bool flip_horizontal = false;
    bool use_mjpg = true;
    int warmup_frames = 12;
};

struct TagSpec {
    std::string mode = "auto";   // auto | family | id
    std::string family = "AprilTag 36h11";
    int id = -1;
};

struct RedThreshold {
    int hue_low_1 = 0;
    int hue_high_1 = 12;
    int hue_low_2 = 170;
    int hue_high_2 = 180;
    int sat_min = 100;
    int val_min = 70;
    double ratio_trigger = 0.18;
    int pixel_mean_r_min = 90;
    int cooldown_frames = 20;
};

struct DeployBehavior {
    bool save_trigger_images = true;
    std::string save_dir = "captures";
    bool draw_debug = true;
};

struct CalibrationData {
    bool valid = false;
    std::string family = "AprilTag 36h11";
    int id = -1;
    int source_frame_width = 0;
    int source_frame_height = 0;
    int warp_width = 0;
    int warp_height = 0;
    std::array<double, 9> H{};
    std::array<std::array<float,2>,4> tag_corners{};
    RoiNorm red_roi;
    RoiNorm image_roi;
};

struct AppConfig {
    CameraProfile camera;
    TagSpec tag;
    RedThreshold red;
    DeployBehavior deploy;
    CalibrationData calibration;
};

struct Detection {
    std::string family;
    int id = -1;
    std::vector<cv::Point2f> corners;
};

struct ProbeResult {
    int camera_index = -1;
    std::string backend;
    int req_w = 0, req_h = 0, req_fps = 0;
    int act_w = 0, act_h = 0;
    double fps_measured = 0.0;
    bool open_ok = false;
    bool read_ok = false;
    bool stable = false;
    double mean_luma = 0.0;
    double luma_std = 0.0;
    std::string note;
};

struct SelectionState {
    bool active = false;
    bool finished = false;
    bool dragging = false;
    cv::Point start{};
    cv::Point current{};
    RoiNorm* target = nullptr;
    int ref_w = 1;
    int ref_h = 1;
    std::string label;
};

// ----------------------------- JSON ----------------------------------------
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(RoiNorm, x, y, w, h)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CameraProfile, index, width, height, fps, backend, flip_horizontal, use_mjpg, warmup_frames)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TagSpec, mode, family, id)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(RedThreshold, hue_low_1, hue_high_1, hue_low_2, hue_high_2, sat_min, val_min, ratio_trigger, pixel_mean_r_min, cooldown_frames)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DeployBehavior, save_trigger_images, save_dir, draw_debug)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CalibrationData, valid, family, id, source_frame_width, source_frame_height, warp_width, warp_height, H, tag_corners, red_roi, image_roi)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AppConfig, camera, tag, red, deploy, calibration)

static AppConfig defaultConfig() {
    AppConfig c;
    c.camera = CameraProfile{};
    c.tag = TagSpec{};
    c.red = RedThreshold{};
    c.deploy = DeployBehavior{};
    c.calibration = CalibrationData{};
    return c;
}

static AppConfig loadConfig(const fs::path& path) {
    if (!fs::exists(path)) {
        AppConfig c = defaultConfig();
        ensureDir(path.parent_path());
        std::ofstream(path) << json(c).dump(2) << "\n";
        return c;
    }
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open config: " + path.string());
    json j; f >> j;
    return j.get<AppConfig>();
}

static void saveConfig(const AppConfig& cfg, const fs::path& path) {
    ensureDir(path.parent_path());
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot write config: " + path.string());
    f << json(cfg).dump(2) << "\n";
}

// ----------------------------- CLI -----------------------------------------
struct Cli {
    std::string mode = "help"; // probe | calibrate | deploy
    fs::path config = "config/system_config.json";
    std::optional<int> camera_index;
    std::optional<int> width;
    std::optional<int> height;
    std::optional<int> fps;
    std::optional<std::string> tag_family;
    std::optional<int> tag_id;
    std::optional<bool> flip;
};

static Cli parseArgs(int argc, char** argv) {
    Cli cli;
    if (argc >= 2) cli.mode = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto req = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--config") cli.config = req();
        else if (a == "--camera") cli.camera_index = std::stoi(req());
        else if (a == "--width") cli.width = std::stoi(req());
        else if (a == "--height") cli.height = std::stoi(req());
        else if (a == "--fps") cli.fps = std::stoi(req());
        else if (a == "--tag-family") cli.tag_family = req();
        else if (a == "--tag-id") cli.tag_id = std::stoi(req());
        else if (a == "--flip") cli.flip = true;
        else if (a == "--no-flip") cli.flip = false;
        else if (a == "--help" || a == "-h") cli.mode = "help";
        else throw std::runtime_error("unknown argument: " + a);
    }
    return cli;
}

// ----------------------------- camera --------------------------------------
static int backendFromName(const std::string& s) {
    const std::string b = lower(s);
    if (b == "any") return cv::CAP_ANY;
#ifdef _WIN32
    if (b == "msmf") return cv::CAP_MSMF;
    if (b == "dshow") return cv::CAP_DSHOW;
#else
    if (b == "v4l2") return cv::CAP_V4L2;
    if (b == "gstreamer") return cv::CAP_GSTREAMER;
#endif
    return cv::CAP_ANY;
}

static std::string backendName(int backend) {
#ifdef _WIN32
    if (backend == cv::CAP_MSMF) return "MSMF";
    if (backend == cv::CAP_DSHOW) return "DSHOW";
#endif
#ifdef __linux__
    if (backend == cv::CAP_V4L2) return "V4L2";
    if (backend == cv::CAP_GSTREAMER) return "GSTREAMER";
#endif
    return "ANY";
}

static cv::VideoCapture openCamera(const CameraProfile& cp) {
    int backend = backendFromName(cp.backend);
    cv::VideoCapture cap(cp.index, backend);
    if (!cap.isOpened() && backend != cv::CAP_ANY) {
        cap.open(cp.index, cv::CAP_ANY);
    }
    if (!cap.isOpened()) throw std::runtime_error("failed to open camera index " + std::to_string(cp.index));

    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    if (cp.use_mjpg) cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, cp.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cp.height);
    cap.set(cv::CAP_PROP_FPS, cp.fps);
    for (int i = 0; i < std::max(0, cp.warmup_frames); ++i) {
        cv::Mat throwaway;
        cap.read(throwaway);
    }
    return cap;
}

static cv::Rect roiToRect(const RoiNorm& r, int W, int H) {
    int x = (int)std::round(clamp01(r.x) * W);
    int y = (int)std::round(clamp01(r.y) * H);
    int w = (int)std::round(clamp01(r.w) * W);
    int h = (int)std::round(clamp01(r.h) * H);
    x = std::clamp(x, 0, std::max(0, W - 1));
    y = std::clamp(y, 0, std::max(0, H - 1));
    w = std::clamp(w, 1, W - x);
    h = std::clamp(h, 1, H - y);
    return cv::Rect(x, y, w, h);
}

static RoiNorm rectToNorm(const cv::Rect& r, int W, int H) {
    return RoiNorm{
        W > 0 ? (double)r.x / W : 0.0,
        H > 0 ? (double)r.y / H : 0.0,
        W > 0 ? (double)r.width / W : 0.0,
        H > 0 ? (double)r.height / H : 0.0
    };
}

// ----------------------------- tags ----------------------------------------
static const std::map<std::string, int> kDictMap = {
    {"AprilTag 16h5", cv::aruco::DICT_APRILTAG_16h5},
    {"AprilTag 25h9", cv::aruco::DICT_APRILTAG_25h9},
    {"AprilTag 36h10", cv::aruco::DICT_APRILTAG_36h10},
    {"AprilTag 36h11", cv::aruco::DICT_APRILTAG_36h11},
};

struct DetectorBankEntry {
    std::string name;
    cv::Ptr<cv::aruco::Dictionary> dict;
    cv::Ptr<cv::aruco::DetectorParameters> params;
};

static std::vector<DetectorBankEntry> buildDetectorBank(const TagSpec& spec) {
    auto params = cv::aruco::DetectorParameters::create();
    params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_APRILTAG;
    std::vector<DetectorBankEntry> out;
    if (lower(spec.mode) == "auto") {
        for (auto& [name, id] : kDictMap) out.push_back({name, cv::aruco::getPredefinedDictionary(id), params});
    } else {
        auto it = kDictMap.find(spec.family);
        if (it == kDictMap.end()) throw std::runtime_error("unsupported family: " + spec.family);
        out.push_back({it->first, cv::aruco::getPredefinedDictionary(it->second), params});
    }
    return out;
}

static std::vector<Detection> detectTags(const cv::Mat& bgr, const TagSpec& spec, const std::vector<DetectorBankEntry>& bank) {
    cv::Mat gray;
    if (bgr.channels() == 3) cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    else gray = bgr;
    std::vector<Detection> out;
    for (const auto& d : bank) {
        std::vector<std::vector<cv::Point2f>> corners, rejected;
        std::vector<int> ids;
        cv::aruco::detectMarkers(gray, d.dict, corners, ids, d.params, rejected);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (lower(spec.mode) == "id" && spec.id >= 0 && ids[i] != spec.id) continue;
            out.push_back(Detection{d.name, ids[i], corners[i]});
        }
    }
    return out;
}

static std::optional<Detection> chooseDetection(const std::vector<Detection>& ds, const TagSpec& spec) {
    if (ds.empty()) return std::nullopt;
    if (lower(spec.mode) == "id") {
        for (const auto& d : ds) if (d.id == spec.id && d.family == spec.family) return d;
        for (const auto& d : ds) if (d.id == spec.id) return d;
    }
    if (lower(spec.mode) == "family") {
        for (const auto& d : ds) if (d.family == spec.family) return d;
    }
    return ds.front();
}

// ----------------------------- homography ----------------------------------
static CalibrationData buildCalibrationData(const cv::Mat& frame, const Detection& det) {
    const auto& c = det.corners;
    if (c.size() != 4) throw std::runtime_error("need exactly 4 tag corners");
    double top = cv::norm(c[1] - c[0]);
    double right = cv::norm(c[2] - c[1]);
    double bottom = cv::norm(c[2] - c[3]);
    double left = cv::norm(c[3] - c[0]);
    int side = std::max(32, (int)std::round((top + right + bottom + left) * 0.25));

    std::vector<cv::Point2f> dst = {{0,0},{(float)(side-1),0},{(float)(side-1),(float)(side-1)},{0,(float)(side-1)}};
    cv::Mat H_tag = cv::getPerspectiveTransform(c, dst);

    std::vector<cv::Point2f> frameCorners = {{0,0},{(float)(frame.cols-1),0},{(float)(frame.cols-1),(float)(frame.rows-1)},{0,(float)(frame.rows-1)}};
    std::vector<cv::Point2f> proj;
    cv::perspectiveTransform(frameCorners, proj, H_tag);
    float minx = proj[0].x, miny = proj[0].y, maxx = proj[0].x, maxy = proj[0].y;
    for (const auto& p : proj) {
        minx = std::min(minx, p.x); miny = std::min(miny, p.y);
        maxx = std::max(maxx, p.x); maxy = std::max(maxy, p.y);
    }
    double tx = -std::min(0.0f, minx);
    double ty = -std::min(0.0f, miny);
    cv::Mat T = (cv::Mat_<double>(3,3) << 1,0,tx, 0,1,ty, 0,0,1);
    cv::Mat H = T * H_tag;

    CalibrationData out;
    out.valid = true;
    out.family = det.family;
    out.id = det.id;
    out.source_frame_width = frame.cols;
    out.source_frame_height = frame.rows;
    out.warp_width = std::max(side, (int)std::ceil(maxx + tx));
    out.warp_height = std::max(side, (int)std::ceil(maxy + ty));
    for (int r = 0, k = 0; r < 3; ++r) for (int c0 = 0; c0 < 3; ++c0, ++k) out.H[k] = H.at<double>(r,c0);
    for (int i = 0; i < 4; ++i) out.tag_corners[i] = {c[i].x, c[i].y};
    out.red_roi = RoiNorm{0.20, 0.45, 0.12, 0.12};
    out.image_roi = RoiNorm{0.30, 0.25, 0.40, 0.40};
    return out;
}

static cv::Mat HfromCalibration(const CalibrationData& c) {
    cv::Mat H(3,3,CV_64F);
    for (int r = 0, k = 0; r < 3; ++r) for (int cc = 0; cc < 3; ++cc, ++k) H.at<double>(r,cc) = c.H[k];
    return H;
}

static cv::Mat warpWithCalibration(const cv::Mat& frame, const CalibrationData& c) {
    cv::Mat warped;
    cv::warpPerspective(frame, warped, HfromCalibration(c), cv::Size(c.warp_width, c.warp_height), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return warped;
}

// ----------------------------- drawing -------------------------------------
static void drawDetection(cv::Mat& img, const Detection& det, const cv::Scalar& color = {0,255,0}) {
    std::vector<cv::Point> pts;
    for (const auto& p : det.corners) pts.emplace_back((int)std::round(p.x), (int)std::round(p.y));
    const cv::Point* ptr = pts.data();
    int n = (int)pts.size();
    cv::polylines(img, &ptr, &n, 1, true, color, 2, cv::LINE_AA);
    for (int i = 0; i < (int)pts.size(); ++i) {
        cv::circle(img, pts[i], 4, {0,0,255}, -1, cv::LINE_AA);
        cv::putText(img, std::to_string(i), pts[i] + cv::Point(6,-6), cv::FONT_HERSHEY_SIMPLEX, 0.5, {255,255,255}, 1, cv::LINE_AA);
    }
    cv::putText(img, det.family + " id=" + std::to_string(det.id), pts.front() + cv::Point(0,-10), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);
}

static void drawNormalizedRoi(cv::Mat& img, const RoiNorm& r, const cv::Scalar& color, const std::string& label) {
    cv::Rect rr = roiToRect(r, img.cols, img.rows);
    cv::rectangle(img, rr, color, 2, cv::LINE_AA);
    cv::putText(img, label, rr.tl() + cv::Point(4,18), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);
}

static cv::Rect normalizedRectFromPoints(cv::Point a, cv::Point b, int W, int H) {
    int x0 = std::clamp(std::min(a.x, b.x), 0, std::max(0,W-1));
    int y0 = std::clamp(std::min(a.y, b.y), 0, std::max(0,H-1));
    int x1 = std::clamp(std::max(a.x, b.x), 1, W);
    int y1 = std::clamp(std::max(a.y, b.y), 1, H);
    return cv::Rect(x0, y0, std::max(1, x1 - x0), std::max(1, y1 - y0));
}

// ----------------------------- ROI mouse -----------------------------------
static SelectionState g_selection;
static void roiMouseCb(int event, int x, int y, int, void*) {
    if (!g_selection.active) return;
    if (event == cv::EVENT_LBUTTONDOWN) {
        g_selection.dragging = true;
        g_selection.start = {x,y};
        g_selection.current = {x,y};
    } else if (event == cv::EVENT_MOUSEMOVE && g_selection.dragging) {
        g_selection.current = {x,y};
    } else if (event == cv::EVENT_LBUTTONUP && g_selection.dragging) {
        g_selection.current = {x,y};
        g_selection.dragging = false;
        if (g_selection.target) {
            cv::Rect r = normalizedRectFromPoints(g_selection.start, g_selection.current, g_selection.ref_w, g_selection.ref_h);
            *g_selection.target = rectToNorm(r, g_selection.ref_w, g_selection.ref_h);
        }
        g_selection.finished = true;
        g_selection.active = false;
    }
}

static void beginSelection(RoiNorm* target, int w, int h, const std::string& label) {
    g_selection = SelectionState{};
    g_selection.active = true;
    g_selection.target = target;
    g_selection.ref_w = w;
    g_selection.ref_h = h;
    g_selection.label = label;
}

// ----------------------------- probe ---------------------------------------
static std::vector<int> candidateBackends() {
    std::vector<int> out;
#ifdef _WIN32
    out = {cv::CAP_MSMF, cv::CAP_DSHOW, cv::CAP_ANY};
#else
    out = {cv::CAP_V4L2, cv::CAP_GSTREAMER, cv::CAP_ANY};
#endif
    return out;
}

static ProbeResult probeOne(int camIdx, int backend, int reqW, int reqH, int reqFps, bool useMjpg) {
    ProbeResult r;
    r.camera_index = camIdx;
    r.backend = backendName(backend);
    r.req_w = reqW; r.req_h = reqH; r.req_fps = reqFps;

    cv::VideoCapture cap(camIdx, backend);
    if (!cap.isOpened()) {
        r.note = "open failed";
        return r;
    }
    r.open_ok = true;
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    if (useMjpg) cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, reqW);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, reqH);
    cap.set(cv::CAP_PROP_FPS, reqFps);

    for (int i = 0; i < 8; ++i) { cv::Mat tmp; cap.read(tmp); }

    r.act_w = (int)std::round(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    r.act_h = (int)std::round(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::vector<double> lumas;
    const int N = 30;
    int okFrames = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
        cv::Mat f, gray;
        if (!cap.read(f) || f.empty()) continue;
        okFrames++;
        cv::cvtColor(f, gray, cv::COLOR_BGR2GRAY);
        lumas.push_back(cv::mean(gray)[0]);
    }
    auto t1 = std::chrono::steady_clock::now();
    r.read_ok = okFrames > N / 2;
    double sec = std::chrono::duration<double>(t1 - t0).count();
    r.fps_measured = sec > 0 ? okFrames / sec : 0.0;
    if (!lumas.empty()) {
        r.mean_luma = std::accumulate(lumas.begin(), lumas.end(), 0.0) / lumas.size();
        double acc = 0.0;
        for (double v : lumas) acc += (v - r.mean_luma) * (v - r.mean_luma);
        r.luma_std = std::sqrt(acc / lumas.size());
    }
    r.stable = r.read_ok && r.fps_measured >= reqFps * 0.55;
    if (!r.stable) r.note = "low measured fps or unstable reads";
    return r;
}

static void writeProbeReport(const std::vector<ProbeResult>& results, const fs::path& outDir) {
    ensureDir(outDir);
    const fs::path jsonPath = outDir / ("camera_probe_" + nowStamp() + ".json");
    const fs::path csvPath = outDir / ("camera_probe_" + nowStamp() + ".csv");

    json arr = json::array();
    for (const auto& r : results) {
        arr.push_back({
            {"camera_index", r.camera_index}, {"backend", r.backend},
            {"requested_width", r.req_w}, {"requested_height", r.req_h}, {"requested_fps", r.req_fps},
            {"actual_width", r.act_w}, {"actual_height", r.act_h}, {"measured_fps", r.fps_measured},
            {"open_ok", r.open_ok}, {"read_ok", r.read_ok}, {"stable", r.stable},
            {"mean_luma", r.mean_luma}, {"luma_std", r.luma_std}, {"note", r.note}
        });
    }
    std::ofstream(jsonPath) << arr.dump(2) << "\n";

    std::ofstream csv(csvPath);
    csv << "camera_index,backend,req_w,req_h,req_fps,act_w,act_h,fps_measured,open_ok,read_ok,stable,mean_luma,luma_std,note\n";
    for (const auto& r : results) {
        csv << r.camera_index << ',' << r.backend << ',' << r.req_w << ',' << r.req_h << ',' << r.req_fps << ','
            << r.act_w << ',' << r.act_h << ',' << r.fps_measured << ',' << r.open_ok << ',' << r.read_ok << ','
            << r.stable << ',' << r.mean_luma << ',' << r.luma_std << ',' << '"' << r.note << '"' << "\n";
    }
    std::cout << "Probe reports saved to:\n  " << jsonPath << "\n  " << csvPath << "\n";
}

static void runProbe(AppConfig cfg) {
    std::vector<int> cams = {0,1,2,3,4};
    std::vector<std::pair<int,int>> sizes = {{320,240},{640,480},{1280,720},{1920,1080}};
    std::vector<int> fpss = {15, 30, 60};
    std::vector<ProbeResult> results;

    for (int camIdx : cams) {
        for (int backend : candidateBackends()) {
            for (auto [w,h] : sizes) {
                for (int fps : fpss) {
                    ProbeResult r = probeOne(camIdx, backend, w, h, fps, cfg.camera.use_mjpg);
                    if (r.open_ok) {
                        results.push_back(r);
                        std::cout << "cam=" << camIdx << " backend=" << r.backend
                                  << " req=" << w << "x" << h << "@" << fps
                                  << " act=" << r.act_w << "x" << r.act_h
                                  << " fps_meas=" << std::fixed << std::setprecision(1) << r.fps_measured
                                  << " stable=" << r.stable << "\n";
                    }
                }
            }
        }
    }

    writeProbeReport(results, "reports");

    auto bestIt = std::max_element(results.begin(), results.end(), [](const ProbeResult& a, const ProbeResult& b){
        auto score = [](const ProbeResult& r) {
            return (r.stable ? 1000.0 : 0.0) + r.fps_measured + 0.001 * r.act_w * r.act_h;
        };
        return score(a) < score(b);
    });
    if (bestIt != results.end()) {
        cfg.camera.index = bestIt->camera_index;
        cfg.camera.width = bestIt->act_w > 0 ? bestIt->act_w : bestIt->req_w;
        cfg.camera.height = bestIt->act_h > 0 ? bestIt->act_h : bestIt->req_h;
        cfg.camera.fps = std::max(1, (int)std::round(bestIt->fps_measured));
        cfg.camera.backend = bestIt->backend;
        std::cout << "Best camera profile candidate: camera=" << cfg.camera.index
                  << " " << cfg.camera.width << "x" << cfg.camera.height << " @~" << cfg.camera.fps
                  << " backend=" << cfg.camera.backend << "\n";
    } else {
        std::cout << "No camera profiles passed open/read probing.\n";
    }
}

// ----------------------------- calibration ---------------------------------
static void overlayCalibrationHelp(cv::Mat& raw, cv::Mat& warp, const std::string& status) {
    cv::putText(raw, "[L] lock transform  [R] red ROI  [I] image ROI  [S] save  [Q] quit", {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {30,220,255}, 2, cv::LINE_AA);
    cv::putText(raw, status, {12, raw.rows - 14}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,255}, 2, cv::LINE_AA);
    cv::putText(warp, "Warp view: drag rectangle after pressing R or I", {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {30,220,255}, 2, cv::LINE_AA);
}

static void maybeDrawSelection(cv::Mat& img) {
    if (g_selection.dragging) {
        cv::Rect rr = normalizedRectFromPoints(g_selection.start, g_selection.current, img.cols, img.rows);
        cv::rectangle(img, rr, {255,255,0}, 2, cv::LINE_AA);
        cv::putText(img, g_selection.label, rr.tl() + cv::Point(3,18), cv::FONT_HERSHEY_SIMPLEX, 0.6, {255,255,0}, 2, cv::LINE_AA);
    }
}

static void runCalibrate(AppConfig& cfg, const fs::path& configPath) {
    auto bank = buildDetectorBank(cfg.tag);
    cv::VideoCapture cap = openCamera(cfg.camera);
    cv::namedWindow(kWindowCalRaw, cv::WINDOW_NORMAL);
    cv::namedWindow(kWindowCalWarp, cv::WINDOW_NORMAL);
    cv::setMouseCallback(kWindowCalWarp, roiMouseCb);

    CalibrationData liveCal = cfg.calibration;
    bool locked = false;
    std::string status = "show tag in view";

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) throw std::runtime_error("camera read failed during calibration");
        if (cfg.camera.flip_horizontal) cv::flip(frame, frame, 1);

        auto ds = detectTags(frame, cfg.tag, bank);
        auto chosen = chooseDetection(ds, cfg.tag);

        cv::Mat rawDisp = frame.clone();
        cv::Mat warpDisp(std::max(240, frame.rows), std::max(320, frame.cols), frame.type(), cv::Scalar(32,32,32));

        if (!locked && chosen) {
            drawDetection(rawDisp, *chosen);
            liveCal = buildCalibrationData(frame, *chosen);
            status = "tag detected - press L to lock transform";
        }

        if (locked && liveCal.valid) {
            warpDisp = warpWithCalibration(frame, liveCal);
            drawNormalizedRoi(warpDisp, liveCal.red_roi, {0,0,255}, "red_roi");
            drawNormalizedRoi(warpDisp, liveCal.image_roi, {0,255,255}, "image_roi");
            maybeDrawSelection(warpDisp);
        } else if (!chosen) {
            status = "no valid tag detected";
        }

        overlayCalibrationHelp(rawDisp, warpDisp, status);
        cv::imshow(kWindowCalRaw, rawDisp);
        cv::imshow(kWindowCalWarp, warpDisp);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) break;
        if (key == 'l' && chosen) {
            liveCal = buildCalibrationData(frame, *chosen);
            locked = true;
            status = "transform locked; now press R then drag red ROI, I then drag image ROI";
        }
        if (locked && liveCal.valid && key == 'r') {
            beginSelection(&liveCal.red_roi, liveCal.warp_width, liveCal.warp_height, "red_roi");
            status = "drag red ROI on warp view";
        }
        if (locked && liveCal.valid && key == 'i') {
            beginSelection(&liveCal.image_roi, liveCal.warp_width, liveCal.warp_height, "image_roi");
            status = "drag image ROI on warp view";
        }
        if (locked && liveCal.valid && key == 's') {
            cfg.calibration = liveCal;
            cfg.calibration.valid = true;
            saveConfig(cfg, configPath);
            status = "saved calibration to " + configPath.string();
            std::cout << status << "\n";
        }
    }

    cap.release();
    cv::destroyWindow(kWindowCalRaw);
    cv::destroyWindow(kWindowCalWarp);
}

// ----------------------------- deploy --------------------------------------
static double computeRedRatio(const cv::Mat& roiBgr, const RedThreshold& t, double& meanRed) {
    cv::Mat hsv;
    cv::cvtColor(roiBgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat m1, m2, mask;
    cv::inRange(hsv, cv::Scalar(t.hue_low_1, t.sat_min, t.val_min), cv::Scalar(t.hue_high_1, 255, 255), m1);
    cv::inRange(hsv, cv::Scalar(t.hue_low_2, t.sat_min, t.val_min), cv::Scalar(t.hue_high_2, 255, 255), m2);
    cv::bitwise_or(m1, m2, mask);
    meanRed = cv::mean(roiBgr)[2];
    return (double)cv::countNonZero(mask) / std::max(1, mask.rows * mask.cols);
}

static void runDeploy(const AppConfig& cfg) {
    if (!cfg.calibration.valid) throw std::runtime_error("config calibration is invalid; run calibrate first");
    cv::VideoCapture cap = openCamera(cfg.camera);
    ensureDir(cfg.deploy.save_dir);
    cv::namedWindow(kWindowDeployRaw, cv::WINDOW_NORMAL);
    cv::namedWindow(kWindowDeployWarp, cv::WINDOW_NORMAL);

    int cooldown = 0;
    int triggerCount = 0;
    auto tPrev = std::chrono::steady_clock::now();
    double fpsEma = 0.0;

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) throw std::runtime_error("camera read failed during deploy");
        if (cfg.camera.flip_horizontal) cv::flip(frame, frame, 1);

        auto tNow = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(tNow - tPrev).count();
        tPrev = tNow;
        double fps = dt > 0 ? 1.0 / dt : 0.0;
        fpsEma = fpsEma <= 0 ? fps : (0.92 * fpsEma + 0.08 * fps);

        cv::Mat rawDisp = frame.clone();
        cv::Mat warped = warpWithCalibration(frame, cfg.calibration);
        cv::Rect redRect = roiToRect(cfg.calibration.red_roi, warped.cols, warped.rows);
        cv::Rect imgRect = roiToRect(cfg.calibration.image_roi, warped.cols, warped.rows);
        cv::Mat redPatch = warped(redRect);
        cv::Mat imgPatch = warped(imgRect);

        double meanRed = 0.0;
        double redRatio = computeRedRatio(redPatch, cfg.red, meanRed);
        bool armed = cooldown <= 0;
        bool triggered = armed && redRatio >= cfg.red.ratio_trigger && meanRed >= cfg.red.pixel_mean_r_min;
        if (triggered) {
            triggerCount++;
            cooldown = cfg.red.cooldown_frames;
            if (cfg.deploy.save_trigger_images) {
                std::string base = nowStamp() + "_trig" + std::to_string(triggerCount);
                cv::imwrite((fs::path(cfg.deploy.save_dir) / (base + "_raw.jpg")).string(), frame);
                cv::imwrite((fs::path(cfg.deploy.save_dir) / (base + "_warp.jpg")).string(), warped);
                cv::imwrite((fs::path(cfg.deploy.save_dir) / (base + "_image_roi.jpg")).string(), imgPatch);
            }
        }
        cooldown = std::max(0, cooldown - 1);

        if (cfg.deploy.draw_debug) {
            drawNormalizedRoi(warped, cfg.calibration.red_roi, triggered ? cv::Scalar(0,0,255) : cv::Scalar(80,80,255), "red_roi");
            drawNormalizedRoi(warped, cfg.calibration.image_roi, {0,255,255}, "image_roi");
            std::ostringstream oss;
            oss << "fps " << std::fixed << std::setprecision(1) << fpsEma
                << " red_ratio " << std::setprecision(3) << redRatio
                << " meanR " << std::setprecision(1) << meanRed
                << " trigger_count " << triggerCount
                << (triggered ? " TRIGGER" : "");
            cv::putText(warped, oss.str(), {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.7, triggered ? cv::Scalar(0,0,255) : cv::Scalar(0,255,0), 2, cv::LINE_AA);
            cv::putText(rawDisp, "[Q] quit | deploy mode", {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {30,220,255}, 2, cv::LINE_AA);
        }

        cv::imshow(kWindowDeployRaw, rawDisp);
        cv::imshow(kWindowDeployWarp, warped);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) break;
    }

    cap.release();
    cv::destroyWindow(kWindowDeployRaw);
    cv::destroyWindow(kWindowDeployWarp);
}

// ----------------------------- help ----------------------------------------
static void printHelp() {
    std::cout
        << "camera_combo_app modes:\n"
        << "  probe     : scan camera/resolution/fps combinations and write health reports\n"
        << "  calibrate : detect AprilTag, lock homography, select red ROI and image ROI, save config\n"
        << "  deploy    : load config, warp each frame, monitor red ROI, capture image ROI on trigger\n\n"
        << "Usage:\n"
        << "  camera_combo_app probe --config config/system_config.json\n"
        << "  camera_combo_app calibrate --config config/system_config.json\n"
        << "  camera_combo_app deploy --config config/system_config.json\n\n"
        << "Optional overrides:\n"
        << "  --camera N --width W --height H --fps F --tag-family \"AprilTag 36h11\" --tag-id ID --flip --no-flip\n";
}

// ----------------------------- main ----------------------------------------
int main(int argc, char** argv) {
    try {
        Cli cli = parseArgs(argc, argv);
        if (cli.mode == "help" || cli.mode.empty()) {
            printHelp();
            return 0;
        }

        AppConfig cfg = loadConfig(cli.config);
        if (cli.camera_index) cfg.camera.index = *cli.camera_index;
        if (cli.width) cfg.camera.width = *cli.width;
        if (cli.height) cfg.camera.height = *cli.height;
        if (cli.fps) cfg.camera.fps = *cli.fps;
        if (cli.tag_family) { cfg.tag.family = *cli.tag_family; if (lower(cfg.tag.mode) != "auto") cfg.tag.mode = "family"; }
        if (cli.tag_id) { cfg.tag.id = *cli.tag_id; cfg.tag.mode = "id"; }
        if (cli.flip) cfg.camera.flip_horizontal = *cli.flip;

        if (cli.mode == "probe") {
            runProbe(cfg);
        } else if (cli.mode == "calibrate") {
            runCalibrate(cfg, cli.config);
        } else if (cli.mode == "deploy") {
            runDeploy(cfg);
        } else {
            throw std::runtime_error("unknown mode: " + cli.mode);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
