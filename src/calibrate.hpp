#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#if __has_include(<opencv2/aruco.hpp>)
#include <opencv2/aruco.hpp>
#define VISION_APP_HAS_ARUCO 1
#else
#define VISION_APP_HAS_ARUCO 0
#endif

namespace vision_app {

struct RoiRatio {
    double x = 0.10, y = 0.10, w = 0.20, h = 0.20;
};
struct RoiConfig {
    RoiRatio red_roi{0.10,0.10,0.20,0.20};
    RoiRatio image_roi{0.35,0.15,0.45,0.45};
};

struct AprilTagConfig {
    std::string family = "auto"; // auto|16|25|36
    int target_id = 0;
    bool require_target_id = true;
    bool manual_lock_only = true;
    int lock_frames = 8;
};

struct AprilTagDetection {
    bool found = false;
    std::string family;
    int id = -1;
    std::array<cv::Point2f, 4> corners{};
    cv::Point2f center{};
    float score = 0.0f;
};

struct WarpPackage {
    bool valid = false;
    std::string family;
    int id = -1;
    int src_w = 0;
    int src_h = 0;
    int warp_w = 0;
    int warp_h = 0;
    double soft_limit = 900.0;
    cv::Mat H;           // 3x3 CV_64F
    cv::Mat Hinv;        // 3x3 CV_64F
    cv::Mat map1;        // fixed-point or float map
    cv::Mat map2;
    cv::Mat valid_mask;  // CV_8U
};

static inline double clamp01(double v) { return std::max(0.0, std::min(1.0, v)); }
static inline bool validate_roi(RoiRatio& r) {
    r.x = clamp01(r.x); r.y = clamp01(r.y); r.w = clamp01(r.w); r.h = clamp01(r.h);
    if (r.w <= 0.0 || r.h <= 0.0) return false;
    if (r.x + r.w > 1.0) r.w = 1.0 - r.x;
    if (r.y + r.h > 1.0) r.h = 1.0 - r.y;
    return r.w > 0.0 && r.h > 0.0;
}
static inline cv::Rect roi_to_rect(const RoiRatio& rr, const cv::Size& sz) {
    RoiRatio r = rr; validate_roi(r);
    int x = std::clamp(static_cast<int>(std::round(r.x * sz.width)), 0, std::max(0, sz.width - 1));
    int y = std::clamp(static_cast<int>(std::round(r.y * sz.height)), 0, std::max(0, sz.height - 1));
    int w = std::max(1, static_cast<int>(std::round(r.w * sz.width)));
    int h = std::max(1, static_cast<int>(std::round(r.h * sz.height)));
    if (x + w > sz.width) w = std::max(1, sz.width - x);
    if (y + h > sz.height) h = std::max(1, sz.height - y);
    return {x,y,w,h};
}

static inline std::array<cv::Point2f,4> canonical_square(float side) {
    const float s = side - 1.0f;
    return { cv::Point2f(0,0), cv::Point2f(s,0), cv::Point2f(s,s), cv::Point2f(0,s) };
}

#if VISION_APP_HAS_ARUCO
static inline bool detect_family_once(const cv::Mat& gray, int dict_id, const std::string& family_name, const AprilTagConfig& cfg, AprilTagDetection& out) {
    auto dict = cv::aruco::getPredefinedDictionary(dict_id);
    cv::aruco::DetectorParameters params;
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    cv::aruco::ArucoDetector detector(dict, params);
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners, rejected;
    detector.detectMarkers(gray, corners, ids, rejected);
    if (ids.empty()) return false;
    int best = -1;
    float best_score = -1.0f;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (cfg.require_target_id && ids[i] != cfg.target_id) continue;
        float peri = static_cast<float>(cv::arcLength(corners[i], true));
        if (peri > best_score) { best_score = peri; best = static_cast<int>(i); }
    }
    if (best < 0) return false;
    out = {};
    out.found = true;
    out.family = family_name;
    out.id = ids[best];
    out.score = best_score;
    for (int k = 0; k < 4; ++k) out.corners[k] = corners[best][k];
    out.center = 0.25f * (out.corners[0] + out.corners[1] + out.corners[2] + out.corners[3]);
    return true;
}
#endif

static inline bool detect_apriltag_best(const cv::Mat& frame_bgr, const AprilTagConfig& cfg, AprilTagDetection& out, std::string& err) {
    (void)err;
#if !VISION_APP_HAS_ARUCO
    out = {};
    return false;
#else
    cv::Mat gray;
    if (frame_bgr.channels() == 1) gray = frame_bgr;
    else cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);

    std::vector<std::pair<int,std::string>> tries;
    if (cfg.family == "16") tries.push_back({cv::aruco::DICT_APRILTAG_16h5, "16"});
    else if (cfg.family == "25") tries.push_back({cv::aruco::DICT_APRILTAG_25h9, "25"});
    else if (cfg.family == "36") tries.push_back({cv::aruco::DICT_APRILTAG_36h11, "36"});
    else tries = {{cv::aruco::DICT_APRILTAG_36h11, "36"}, {cv::aruco::DICT_APRILTAG_25h9, "25"}, {cv::aruco::DICT_APRILTAG_16h5, "16"}};

    AprilTagDetection best{};
    for (const auto& t : tries) {
        AprilTagDetection d;
        if (detect_family_once(gray, t.first, t.second, cfg, d)) {
            if (!best.found || d.score > best.score) best = d;
        }
    }
    out = best;
    return out.found;
#endif
}

class TagLocker {
public:
    explicit TagLocker(int req = 8) : required_(std::max(1, req)) {}
    void reset() { count_ = 0; locked_ = false; last_ = {}; }
    bool update(const AprilTagDetection& d) {
        if (!d.found) { reset(); return false; }
        if (!last_.found || last_.id != d.id || last_.family != d.family) count_ = 1;
        else ++count_;
        last_ = d;
        locked_ = (count_ >= required_);
        return locked_;
    }
    bool locked() const { return locked_; }
    AprilTagDetection current() const { return last_; }
private:
    int required_ = 8;
    int count_ = 0;
    bool locked_ = false;
    AprilTagDetection last_{};
};

static inline bool build_warp_package_from_detection(const AprilTagDetection& det,
                                                     const cv::Size& src_size,
                                                     int warp_soft_max,
                                                     WarpPackage& pack,
                                                     std::string& err) {
    if (!det.found) { err = "no detection to build warp"; return false; }
    if (src_size.width <= 0 || src_size.height <= 0) { err = "invalid source size"; return false; }

    const float canonical_side = 1000.0f;
    std::vector<cv::Point2f> src(4), dst(4);
    for (int i = 0; i < 4; ++i) src[i] = det.corners[i];
    auto canon = canonical_square(canonical_side);
    for (int i = 0; i < 4; ++i) dst[i] = canon[i];

    cv::Mat H0 = cv::getPerspectiveTransform(src, dst);
    if (H0.empty()) { err = "getPerspectiveTransform failed"; return false; }

    std::vector<cv::Point2f> image_corners = {
        {0.0f, 0.0f},
        {static_cast<float>(src_size.width - 1), 0.0f},
        {static_cast<float>(src_size.width - 1), static_cast<float>(src_size.height - 1)},
        {0.0f, static_cast<float>(src_size.height - 1)}
    };
    std::vector<cv::Point2f> transformed;
    cv::perspectiveTransform(image_corners, transformed, H0);

    float min_x = transformed[0].x, max_x = transformed[0].x;
    float min_y = transformed[0].y, max_y = transformed[0].y;
    for (const auto& p : transformed) {
        min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
    }
    double bbox_w = std::max(1.0, static_cast<double>(max_x - min_x));
    double bbox_h = std::max(1.0, static_cast<double>(max_y - min_y));
    double scale = 1.0;
    const double max_dim = std::max(bbox_w, bbox_h);
    if (warp_soft_max > 0 && max_dim > static_cast<double>(warp_soft_max)) {
        scale = static_cast<double>(warp_soft_max) / max_dim;
    }
    // hard limit for Pi stability
    if (max_dim * scale > 1200.0) scale *= 1200.0 / (max_dim * scale);

    cv::Mat T = (cv::Mat_<double>(3,3) << 1,0,-min_x, 0,1,-min_y, 0,0,1);
    cv::Mat S = (cv::Mat_<double>(3,3) << scale,0,0, 0,scale,0, 0,0,1);
    cv::Mat H = S * T * H0;
    cv::Mat Hinv = H.inv();
    if (H.empty() || Hinv.empty()) { err = "homography inversion failed"; return false; }

    int out_w = std::max(1, static_cast<int>(std::ceil(bbox_w * scale)));
    int out_h = std::max(1, static_cast<int>(std::ceil(bbox_h * scale)));
    out_w = std::min(out_w, 1200);
    out_h = std::min(out_h, 1200);

    cv::Mat mapx(out_h, out_w, CV_32FC1);
    cv::Mat mapy(out_h, out_w, CV_32FC1);
    cv::Mat valid(out_h, out_w, CV_8UC1, cv::Scalar(0));

    const double h00 = Hinv.at<double>(0,0), h01 = Hinv.at<double>(0,1), h02 = Hinv.at<double>(0,2);
    const double h10 = Hinv.at<double>(1,0), h11 = Hinv.at<double>(1,1), h12 = Hinv.at<double>(1,2);
    const double h20 = Hinv.at<double>(2,0), h21 = Hinv.at<double>(2,1), h22 = Hinv.at<double>(2,2);

    for (int y = 0; y < out_h; ++y) {
        float* mx = mapx.ptr<float>(y);
        float* my = mapy.ptr<float>(y);
        uchar* vm = valid.ptr<uchar>(y);
        for (int x = 0; x < out_w; ++x) {
            const double X = static_cast<double>(x);
            const double Y = static_cast<double>(y);
            const double w = h20*X + h21*Y + h22;
            const double sx = (h00*X + h01*Y + h02) / w;
            const double sy = (h10*X + h11*Y + h12) / w;
            mx[x] = static_cast<float>(sx);
            my[x] = static_cast<float>(sy);
            vm[x] = (sx >= 0.0 && sx < static_cast<double>(src_size.width) && sy >= 0.0 && sy < static_cast<double>(src_size.height)) ? 255 : 0;
        }
    }

    cv::Mat map1, map2;
    cv::convertMaps(mapx, mapy, map1, map2, CV_16SC2, true);

    pack = {};
    pack.valid = true;
    pack.family = det.family;
    pack.id = det.id;
    pack.src_w = src_size.width;
    pack.src_h = src_size.height;
    pack.warp_w = out_w;
    pack.warp_h = out_h;
    pack.soft_limit = warp_soft_max;
    pack.H = H;
    pack.Hinv = Hinv;
    pack.map1 = map1;
    pack.map2 = map2;
    pack.valid_mask = valid;
    return true;
}

static inline bool apply_warp(const cv::Mat& src, const WarpPackage& pack, cv::Mat& warped, cv::Mat& valid_mask) {
    if (!pack.valid || src.empty()) return false;
    cv::remap(src, warped, pack.map1, pack.map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    valid_mask = pack.valid_mask;
    return !warped.empty();
}

static inline cv::Mat compose_preview_with_mask(const cv::Mat& warped, const cv::Mat& valid_mask) {
    cv::Mat out(warped.size(), warped.type(), cv::Scalar(55,55,55));
    if (warped.empty()) return out;
    warped.copyTo(out, valid_mask);
    return out;
}

static inline void draw_detection(cv::Mat& img, const AprilTagDetection& det, bool locked) {
    if (!det.found) return;
    const cv::Scalar color = locked ? cv::Scalar(0,255,0) : cv::Scalar(0,255,255);
    for (int i = 0; i < 4; ++i) cv::line(img, det.corners[i], det.corners[(i+1)%4], color, 2);
    cv::circle(img, det.center, 4, cv::Scalar(255,0,255), -1);
    std::string text = "tag family=" + det.family + " id=" + std::to_string(det.id) + (locked ? " [LOCKED]" : "");
    cv::putText(img, text, {20,30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
}

static inline void draw_rois(cv::Mat& img, const RoiConfig& rois, int selected) {
    const cv::Rect rr = roi_to_rect(rois.red_roi, img.size());
    const cv::Rect ir = roi_to_rect(rois.image_roi, img.size());
    cv::rectangle(img, rr, selected == 1 ? cv::Scalar(0,255,255) : cv::Scalar(0,0,255), 2);
    cv::putText(img, "red_roi", rr.tl() + cv::Point(4,20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);
    cv::rectangle(img, ir, selected == 2 ? cv::Scalar(0,255,255) : cv::Scalar(255,0,0), 2);
    cv::putText(img, "image_roi", ir.tl() + cv::Point(4,20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,0,0), 2);
}

static inline bool save_rois_yaml(const std::string& path, const RoiConfig& rois) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;
    fs << "red_roi" << "{" << "x" << rois.red_roi.x << "y" << rois.red_roi.y << "w" << rois.red_roi.w << "h" << rois.red_roi.h << "}";
    fs << "image_roi" << "{" << "x" << rois.image_roi.x << "y" << rois.image_roi.y << "w" << rois.image_roi.w << "h" << rois.image_roi.h << "}";
    return true;
}

static inline bool load_rois_yaml(const std::string& path, RoiConfig& rois) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    RoiConfig tmp = rois;
    auto rr = fs["red_roi"]; auto ir = fs["image_roi"];
    if (rr.empty() || ir.empty()) return false;
    rr["x"] >> tmp.red_roi.x; rr["y"] >> tmp.red_roi.y; rr["w"] >> tmp.red_roi.w; rr["h"] >> tmp.red_roi.h;
    ir["x"] >> tmp.image_roi.x; ir["y"] >> tmp.image_roi.y; ir["w"] >> tmp.image_roi.w; ir["h"] >> tmp.image_roi.h;
    if (!validate_roi(tmp.red_roi) || !validate_roi(tmp.image_roi)) return false;
    rois = tmp;
    return true;
}

static inline bool save_warp_package(const std::string& path, const WarpPackage& pack) {
    if (!pack.valid) return false;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    cv::FileStorage fs(path, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
    if (!fs.isOpened()) return false;
    fs << "valid" << 1;
    fs << "family" << pack.family;
    fs << "id" << pack.id;
    fs << "src_w" << pack.src_w << "src_h" << pack.src_h;
    fs << "warp_w" << pack.warp_w << "warp_h" << pack.warp_h;
    fs << "soft_limit" << pack.soft_limit;
    fs << "H" << pack.H;
    fs << "Hinv" << pack.Hinv;
    fs << "map1" << pack.map1;
    fs << "map2" << pack.map2;
    fs << "valid_mask" << pack.valid_mask;
    return true;
}

static inline bool load_warp_package(const std::string& path, WarpPackage& pack) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    int valid = 0;
    fs["valid"] >> valid;
    if (!valid) return false;
    pack = {};
    pack.valid = true;
    fs["family"] >> pack.family;
    fs["id"] >> pack.id;
    fs["src_w"] >> pack.src_w; fs["src_h"] >> pack.src_h;
    fs["warp_w"] >> pack.warp_w; fs["warp_h"] >> pack.warp_h;
    fs["soft_limit"] >> pack.soft_limit;
    fs["H"] >> pack.H; fs["Hinv"] >> pack.Hinv;
    fs["map1"] >> pack.map1; fs["map2"] >> pack.map2;
    fs["valid_mask"] >> pack.valid_mask;
    return pack.valid && !pack.map1.empty() && !pack.map2.empty() && !pack.valid_mask.empty();
}

static inline void nudge_roi(RoiRatio& r, char key, double move_step, double size_step) {
    if (key == 'a') r.x -= move_step;
    if (key == 'd') r.x += move_step;
    if (key == 'w') r.y -= move_step;
    if (key == 's') r.y += move_step;
    if (key == 'j') r.w = std::max(0.01, r.w - size_step);
    if (key == 'l') r.w = std::min(1.00, r.w + size_step);
    if (key == 'i') r.h = std::max(0.01, r.h - size_step);
    if (key == 'k') r.h = std::min(1.00, r.h + size_step);
    validate_roi(r);
}

} // namespace vision_app
