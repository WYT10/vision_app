#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace vision_app {

struct AprilTagConfig {
    std::string family = "auto"; // auto | 16 | 25 | 36
    int target_id = 0;
    bool require_target_id = true;
    bool manual_lock_only = true;
    int lock_frames = 4;
};

struct AprilTagDetection {
    bool found = false;
    std::string family;
    int id = -1;
    std::array<cv::Point2f,4> corners{};
    cv::Point2f center{};
};

struct RoiRatio { double x=0.08, y=0.08, w=0.18, h=0.18; };
struct RoiConfig { RoiRatio red_roi; RoiRatio image_roi{0.32,0.10,0.50,0.55}; };

struct WarpPackage {
    bool valid = false;
    cv::Mat H;          // 3x3 CV_64F, source -> warp
    cv::Mat Hinv;       // 3x3 CV_64F, warp -> source
    cv::Mat map1;       // remap maps
    cv::Mat map2;
    cv::Mat valid_mask; // CV_8UC1
    cv::Size src_size;
    cv::Size warp_size;
    std::string family;
    int id = -1;
    double tag_fill_ratio = 0.70;
};

inline bool finite_pt(const cv::Point2f& p) {
    return std::isfinite(p.x) && std::isfinite(p.y);
}

inline double quad_area4(const std::array<cv::Point2f,4>& c) {
    double a = 0.0;
    for (int i = 0; i < 4; ++i) {
        const auto& p = c[i];
        const auto& q = c[(i + 1) % 4];
        a += static_cast<double>(p.x) * q.y - static_cast<double>(q.x) * p.y;
    }
    return 0.5 * std::abs(a);
}

inline std::vector<int> families_from_mode(const std::string& mode) {
    if (mode == "16") return {cv::aruco::DICT_APRILTAG_16h5};
    if (mode == "25") return {cv::aruco::DICT_APRILTAG_25h9};
    if (mode == "36") return {cv::aruco::DICT_APRILTAG_36h11};
    return {cv::aruco::DICT_APRILTAG_36h11, cv::aruco::DICT_APRILTAG_25h9, cv::aruco::DICT_APRILTAG_16h5};
}

inline std::string family_name_from_dict(int dict) {
    switch (dict) {
        case cv::aruco::DICT_APRILTAG_16h5: return "16";
        case cv::aruco::DICT_APRILTAG_25h9: return "25";
        default: return "36";
    }
}

inline bool detect_apriltag_best(const cv::Mat& frame,
                                 const AprilTagConfig& cfg,
                                 AprilTagDetection& det,
                                 std::string& err) {
    det = {};
    if (frame.empty()) { err = "empty frame"; return false; }
    cv::Mat gray;
    if (frame.channels() == 1) gray = frame;
    else cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    cv::aruco::DetectorParameters params;
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    params.minMarkerPerimeterRate = 0.03;
    params.maxMarkerPerimeterRate = 4.0;

    for (const int dict_id : families_from_mode(cfg.family)) {
        auto dict = cv::aruco::getPredefinedDictionary(dict_id);
        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;
        cv::aruco::ArucoDetector detector(dict, params);
        detector.detectMarkers(gray, corners, ids);
        if (ids.empty()) continue;
        int best = 0;
        if (cfg.require_target_id) {
            best = -1;
            for (int i = 0; i < static_cast<int>(ids.size()); ++i) {
                if (ids[i] == cfg.target_id) { best = i; break; }
            }
            if (best < 0) continue;
        }
        det.found = true;
        det.family = family_name_from_dict(dict_id);
        det.id = ids[best];
        for (int k = 0; k < 4; ++k) det.corners[k] = corners[best][k];
        det.center = 0.25f * (det.corners[0] + det.corners[1] + det.corners[2] + det.corners[3]);
        err.clear();
        return true;
    }
    err.clear();
    return true;
}

class TagLocker {
public:
    explicit TagLocker(int required_frames) : required_frames_(std::max(1, required_frames)) {}
    void reset() { hist_.clear(); }
    bool update(const AprilTagDetection& d) {
        if (!d.found) { reset(); return false; }
        if (!hist_.empty() && (hist_.back().id != d.id || hist_.back().family != d.family)) reset();
        hist_.push_back(d);
        while (static_cast<int>(hist_.size()) > required_frames_) hist_.pop_front();
        return static_cast<int>(hist_.size()) >= required_frames_;
    }
private:
    int required_frames_;
    std::deque<AprilTagDetection> hist_;
};

inline RoiRatio clamp_roi(RoiRatio r) {
    auto clamp01 = [](double v){ return std::max(0.0, std::min(1.0, v)); };
    r.x = clamp01(r.x); r.y = clamp01(r.y); r.w = clamp01(r.w); r.h = clamp01(r.h);
    if (r.x + r.w > 1.0) r.w = std::max(0.01, 1.0 - r.x);
    if (r.y + r.h > 1.0) r.h = std::max(0.01, 1.0 - r.y);
    r.w = std::max(0.01, r.w); r.h = std::max(0.01, r.h);
    return r;
}

inline cv::Rect roi_to_rect(const RoiRatio& rr, const cv::Size& sz) {
    RoiRatio r = clamp_roi(rr);
    int x = std::clamp(static_cast<int>(std::round(r.x * sz.width)), 0, std::max(0, sz.width - 1));
    int y = std::clamp(static_cast<int>(std::round(r.y * sz.height)), 0, std::max(0, sz.height - 1));
    int w = std::clamp(static_cast<int>(std::round(r.w * sz.width)), 1, sz.width - x);
    int h = std::clamp(static_cast<int>(std::round(r.h * sz.height)), 1, sz.height - y);
    return {x,y,w,h};
}

inline bool save_rois_yaml(const std::string& path, const RoiConfig& rois) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;
    fs << "red_roi" << "{" << "x" << rois.red_roi.x << "y" << rois.red_roi.y << "w" << rois.red_roi.w << "h" << rois.red_roi.h << "}";
    fs << "image_roi" << "{" << "x" << rois.image_roi.x << "y" << rois.image_roi.y << "w" << rois.image_roi.w << "h" << rois.image_roi.h << "}";
    return true;
}

inline bool load_rois_yaml(const std::string& path, RoiConfig& rois) {
    if (!std::filesystem::exists(path)) return false;
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    auto read_roi = [](const cv::FileNode& n, RoiRatio& r) {
        if (n.empty()) return false;
        n["x"] >> r.x; n["y"] >> r.y; n["w"] >> r.w; n["h"] >> r.h; r = clamp_roi(r); return true;
    };
    read_roi(fs["red_roi"], rois.red_roi);
    read_roi(fs["image_roi"], rois.image_roi);
    return true;
}

inline bool build_centered_warp_package_from_detection(const AprilTagDetection& det,
                                                       const cv::Size& src_size,
                                                       int canvas_side,
                                                       double tag_fill_ratio,
                                                       WarpPackage& pack,
                                                       std::string& err) {
    pack = {};
    if (!det.found) { err = "tag not found"; return false; }
    for (const auto& p : det.corners) {
        if (!finite_pt(p)) { err = "non-finite tag corner"; return false; }
    }
    if (quad_area4(det.corners) < 64.0) { err = "tag quadrilateral too small"; return false; }

    int side = std::max(160, canvas_side);
    side = std::min(side, 1024);
    const double fill = std::clamp(tag_fill_ratio, 0.25, 0.90);
    const double inner = static_cast<double>(side) * fill;
    const double margin = 0.5 * (static_cast<double>(side) - inner);

    std::vector<cv::Point2f> src_quad(4), dst_quad(4);
    for (int i = 0; i < 4; ++i) src_quad[i] = det.corners[i];
    dst_quad[0] = cv::Point2f(static_cast<float>(margin), static_cast<float>(margin));
    dst_quad[1] = cv::Point2f(static_cast<float>(margin + inner), static_cast<float>(margin));
    dst_quad[2] = cv::Point2f(static_cast<float>(margin + inner), static_cast<float>(margin + inner));
    dst_quad[3] = cv::Point2f(static_cast<float>(margin), static_cast<float>(margin + inner));

    cv::Mat H = cv::getPerspectiveTransform(src_quad, dst_quad);
    if (H.empty()) { err = "failed to compute homography"; return false; }
    cv::Mat Hinv = H.inv();
    if (Hinv.empty()) { err = "homography inversion failed"; return false; }

    cv::Mat mapx(side, side, CV_32FC1, cv::Scalar(-1.0f));
    cv::Mat mapy(side, side, CV_32FC1, cv::Scalar(-1.0f));
    cv::Mat valid(side, side, CV_8UC1, cv::Scalar(0));

    const cv::Matx33d Hinvx(
        Hinv.at<double>(0,0), Hinv.at<double>(0,1), Hinv.at<double>(0,2),
        Hinv.at<double>(1,0), Hinv.at<double>(1,1), Hinv.at<double>(1,2),
        Hinv.at<double>(2,0), Hinv.at<double>(2,1), Hinv.at<double>(2,2));

    for (int y = 0; y < side; ++y) {
        for (int x = 0; x < side; ++x) {
            const cv::Vec3d v(static_cast<double>(x), static_cast<double>(y), 1.0);
            const cv::Vec3d p = Hinvx * v;
            const double w = p[2];
            if (!std::isfinite(w) || std::abs(w) < 1e-12) continue;
            const double sx = p[0] / w;
            const double sy = p[1] / w;
            if (std::isfinite(sx) && std::isfinite(sy) &&
                sx >= 0.0 && sy >= 0.0 &&
                sx < static_cast<double>(src_size.width) &&
                sy < static_cast<double>(src_size.height)) {
                mapx.at<float>(y,x) = static_cast<float>(sx);
                mapy.at<float>(y,x) = static_cast<float>(sy);
                valid.at<unsigned char>(y,x) = 255;
            }
        }
    }

    cv::Mat map1, map2;
    cv::convertMaps(mapx, mapy, map1, map2, CV_16SC2);

    pack.valid = true;
    pack.H = H;
    pack.Hinv = Hinv;
    pack.map1 = map1;
    pack.map2 = map2;
    pack.valid_mask = valid;
    pack.src_size = src_size;
    pack.warp_size = {side, side};
    pack.family = det.family;
    pack.id = det.id;
    pack.tag_fill_ratio = fill;
    err.clear();
    return true;
}

inline bool build_warp_package_from_detection(const AprilTagDetection& det,
                                              const cv::Size& src_size,
                                              int warp_soft_max,
                                              WarpPackage& pack,
                                              std::string& err) {
    return build_centered_warp_package_from_detection(det, src_size, warp_soft_max, 0.70, pack, err);
}

inline bool build_warp_package_from_detection(const AprilTagDetection& det,
                                              const cv::Size& src_size,
                                              int warp_soft_max,
                                              double tag_fill_ratio,
                                              WarpPackage& pack,
                                              std::string& err) {
    return build_centered_warp_package_from_detection(det, src_size, warp_soft_max, tag_fill_ratio, pack, err);
}

inline bool apply_warp(const cv::Mat& src, const WarpPackage& pack, cv::Mat& dst, cv::Mat* out_valid = nullptr) {
    if (!pack.valid || src.empty()) return false;
    cv::remap(src, dst, pack.map1, pack.map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255,255,255));
    if (dst.empty()) return false;
    if (out_valid) *out_valid = pack.valid_mask;
    return true;
}

inline bool save_warp_package(const std::string& path, const WarpPackage& pack) {
    if (!pack.valid) return false;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    cv::FileStorage fs(path, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
    if (!fs.isOpened()) return false;
    fs << "family" << pack.family;
    fs << "id" << pack.id;
    fs << "src_w" << pack.src_size.width << "src_h" << pack.src_size.height;
    fs << "warp_w" << pack.warp_size.width << "warp_h" << pack.warp_size.height;
    fs << "tag_fill_ratio" << pack.tag_fill_ratio;
    fs << "H" << pack.H;
    fs << "Hinv" << pack.Hinv;
    fs << "map1" << pack.map1;
    fs << "map2" << pack.map2;
    fs << "valid_mask" << pack.valid_mask;
    return true;
}

inline bool load_warp_package(const std::string& path, WarpPackage& pack) {
    if (!std::filesystem::exists(path)) return false;
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    pack = {};
    int sw=0, sh=0, ww=0, wh=0;
    fs["family"] >> pack.family;
    fs["id"] >> pack.id;
    fs["src_w"] >> sw; fs["src_h"] >> sh;
    fs["warp_w"] >> ww; fs["warp_h"] >> wh;
    if (!fs["tag_fill_ratio"].empty()) fs["tag_fill_ratio"] >> pack.tag_fill_ratio;
    fs["H"] >> pack.H;
    fs["Hinv"] >> pack.Hinv;
    fs["map1"] >> pack.map1;
    fs["map2"] >> pack.map2;
    fs["valid_mask"] >> pack.valid_mask;
    pack.src_size = {sw, sh};
    pack.warp_size = {ww, wh};
    pack.valid = !pack.map1.empty() && !pack.valid_mask.empty();
    return pack.valid;
}

inline void draw_detection_overlay(cv::Mat& img, const AprilTagDetection& det) {
    if (!det.found) return;
    for (int i=0;i<4;++i) cv::line(img, det.corners[i], det.corners[(i+1)%4], cv::Scalar(0,255,255), 2);
    cv::circle(img, det.center, 4, cv::Scalar(255,0,255), -1);
    const std::string txt = "tag family=" + det.family + " id=" + std::to_string(det.id);
    cv::putText(img, txt, {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,255), 2);
}

inline void draw_rois(cv::Mat& img, const RoiConfig& rois, int selected) {
    const cv::Rect rr = roi_to_rect(rois.red_roi, img.size());
    const cv::Rect ir = roi_to_rect(rois.image_roi, img.size());
    cv::rectangle(img, rr, selected==0 ? cv::Scalar(0,0,255) : cv::Scalar(40,40,180), 2);
    cv::putText(img, "red_roi", rr.tl() + cv::Point(4,20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);
    cv::rectangle(img, ir, selected==1 ? cv::Scalar(255,0,0) : cv::Scalar(180,40,40), 2);
    cv::putText(img, "image_roi", ir.tl() + cv::Point(4,20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,0,0), 2);
}

inline void adjust_roi(RoiRatio& r, int key, double move_step, double size_step) {
    if (key=='w') r.y -= move_step;
    if (key=='s') r.y += move_step;
    if (key=='a') r.x -= move_step;
    if (key=='d') r.x += move_step;
    if (key=='j') r.w -= size_step;
    if (key=='l') r.w += size_step;
    if (key=='i') r.h -= size_step;
    if (key=='k') r.h += size_step;
    r = clamp_roi(r);
}

} // namespace vision_app
